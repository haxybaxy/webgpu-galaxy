# GalaxySim Codebase Audit

**Date:** 2026-05-02
**Scope:** `src/*.{hpp,cpp}` plus `CMakeLists.txt`
**Goal:** Identify deficiencies relative to modern C++20 practice, fix opportunities, and concrete next steps.

The codebase is functional and the GPU pipeline design is solid. Most issues below are mechanical / hygiene problems that have accumulated organically. Fixing them will give you a more idiomatic, safer, and more maintainable C++20 codebase without changing the simulation behaviour.

Findings are tagged:

- **🔴 CRITICAL** — correctness bug, leak, or genuinely dangerous; fix first
- **🟠 HIGH** — significant maintainability or robustness problem
- **🟡 MEDIUM** — modernisation / style; clean these up in passes
- **🟢 LOW** — polish and nice-to-haves

---

## Table of Contents

1. [Resource Management & RAII](#1-resource-management--raii)
2. [Modern C++ Language Features Not Used](#2-modern-c-language-features-not-used)
3. [Code Duplication](#3-code-duplication)
4. [Architecture & Design](#4-architecture--design)
5. [Const-Correctness & API Hygiene](#5-const-correctness--api-hygiene)
6. [Type Safety & Numeric Conversions](#6-type-safety--numeric-conversions)
7. [Error Handling](#7-error-handling)
8. [Logging Discipline](#8-logging-discipline)
9. [Magic Numbers & Hidden Constants](#9-magic-numbers--hidden-constants)
10. [Header Hygiene](#10-header-hygiene)
11. [Build System & Tooling](#11-build-system--tooling)
12. [Testing](#12-testing)
13. [WebGPU-Specific Issues](#13-webgpu-specific-issues)
14. [Suggested Refactor Order](#14-suggested-refactor-order)

---

## 1. Resource Management & RAII

This is by far the largest single issue area in the codebase. Most WebGPU handles are managed manually with paired `…Release()` calls. Several handle types are leaked outright.

### 🔴 1.1 `GpuTreeBuilder` has no destructor and no `cleanup()`

`src/gpu_tree_builder.hpp:28-130` declares **11 `WGPUComputePipeline`s, 11 `WGPUBindGroupLayout`s, and 14 cached `WGPUBindGroup`s**. The `.cpp` file contains exactly one place that calls `wgpuBindGroupRelease` (`invalidateBindGroups` at `gpu_tree_builder.cpp:1001-1022`). The pipelines and layouts are **never released** for the lifetime of the program. There is no destructor declared in the header, no `cleanup()` method, and `Application::~Application` never asks the simulation to tear down its tree builder.

**Impact:** every run leaks ~22 GPU objects plus the pipelines they reference. Not catastrophic for a single-shot CLI tool, but it is a real leak and will start mattering if particle count is ever reinitialised in a long-running interactive session.

**Fix:** add `~GpuTreeBuilder()` (or RAII wrappers — see §1.4) that releases every pipeline, every bind group layout, and calls `invalidateBindGroups()`.

### 🔴 1.2 `Simulation` leaks compute pipelines and bind group layouts on destruction

Same problem at `src/simulation.hpp:69-87`: four `WGPUComputePipeline` and four `WGPUBindGroupLayout` members. There is no destructor, no `cleanup()` method, and `Application::~Application` only deletes the GUI/renderer/screenshot helpers. The `Buffer` members are RAII (good), but the pipelines and layouts are not.

### 🔴 1.3 `Application` destructor cleans up out of order

`src/main.cpp:325-342` releases `device_` and `queue_` last. But `gui_`, `renderer_`, `screenshotCapture_`, `simulation_`, and `exporter_` are member objects and will be destroyed by the **default destructor mechanism after `~Application` returns** — at which point `device_` is already released. Currently the manual `gui_.shutdown()` / `renderer_.cleanup()` calls hide this, but `Simulation`, `DiagnosticsCalculator`, and `Exporter` have no equivalent and rely on member destructors. Buffers in `simulation_.positions_` etc. survive into `~Buffer()`, which calls `wgpuBufferRelease` on a buffer whose owning device is already gone.

**Fix:** invert the order — explicitly destroy/clean up everything that holds GPU handles **before** releasing the device. Ideally this is automatic via member-declaration order: declare `device_` first so it's destroyed last.

### 🔴 1.4 Bare WebGPU handles instead of RAII wrappers

Almost every `.cpp` in the project does the dance:

```cpp
WGPUBindGroup bg = wgpuDeviceCreateBindGroup(...);
// ... use bg ...
wgpuBindGroupRelease(bg);
```

This is error-prone (any early return / exception leaks the handle) and clutters the call sites. The codebase already has one RAII wrapper (`wgpu_utils::Buffer`) — extend that pattern.

**Fix:** introduce a small set of `unique_ptr`-style wrappers (or one templated `WgpuHandle<T, Release>`) for `WGPUBindGroup`, `WGPUBindGroupLayout`, `WGPUComputePipeline`, `WGPURenderPipeline`, `WGPUPipelineLayout`, `WGPUShaderModule`, `WGPUTexture`, `WGPUTextureView`, `WGPUSampler`, `WGPUCommandEncoder`, `WGPUCommandBuffer`, `WGPUSurface`, `WGPUDevice`, `WGPUQueue`, `WGPUInstance`, `WGPUAdapter`. A common pattern:

```cpp
template <auto ReleaseFn>
struct WgpuDeleter {
    template <typename H> void operator()(H h) const noexcept {
        if (h) ReleaseFn(h);
    }
};
template <typename H, auto ReleaseFn>
using WgpuUnique = std::unique_ptr<std::remove_pointer_t<H>, WgpuDeleter<ReleaseFn>>;

using BindGroup        = WgpuUnique<WGPUBindGroup,        wgpuBindGroupRelease>;
using ComputePipeline  = WgpuUnique<WGPUComputePipeline,  wgpuComputePipelineRelease>;
// etc.
```

This eliminates ~150 manual release calls across the codebase and makes the code exception-safe.

### 🟠 1.5 `wgpu_utils::Buffer` violates the Rule of Five

`src/wgpu_utils.hpp:28-47` defines a destructor that releases the GPU handle but **does not delete or define copy/move constructors and assignment operators**. The compiler will generate copying that double-releases. Today nothing copies a `Buffer` (you always use members), but this is a foot-gun waiting to happen.

**Fix:**

```cpp
class Buffer {
public:
    Buffer() = default;
    ~Buffer() noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    // ...
};
```

### 🟠 1.6 `Buffer::~Buffer` swallows exceptions silently

`src/wgpu_utils.cpp:273-275`:

```cpp
Buffer::~Buffer() noexcept {
    try { clear(); } catch (...) {}
}
```

`clear()` only calls `wgpuBufferRelease` and zeros members — neither throws. The `try/catch` is dead code that hides the real intent. Mark `clear()` `noexcept` and drop the try/catch.

### 🟠 1.7 `Buffer::upload(void*)` overload creates and releases a queue per call

`src/wgpu_utils.cpp:288-293` calls `wgpuDeviceGetQueue(m_device); … wgpuQueueRelease(queue);` for every upload. Every WebGPU device has exactly one queue; getting/releasing it is cheap but wasteful and clutters the API surface. Either remove this overload (forcing callers to pass the queue, which they always have anyway) or cache the queue in `Buffer`.

### 🟠 1.8 `screenshot.cpp` includes a third-party header *and defines its implementation* in a regular TU

`src/screenshot.cpp:1-2`:

```cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
```

That's correct as long as `screenshot.cpp` is the only TU that does it — but the macro is also pulled into anything that includes `screenshot.hpp` indirectly via the chain `screenshot.hpp → particle_renderer.hpp → wgpu_utils.hpp`. Today nothing else `#define`s it, but if you ever need stb in another file you'll get duplicate-symbol errors. Conventional fix is to put the `#define` inline in a single dedicated `stb_impl.cpp` that does nothing else.

### 🟡 1.9 `globals` in `graphics.cpp` for callbacks

`src/graphics.cpp:6-10` keeps callbacks in file-scope statics:

```cpp
namespace {
    MouseMoveCallback g_mouseMoveCallback;
    ScrollCallback    g_scrollCallback;
    MouseButtonCallback g_mouseButtonCallback;
}
```

This means only one window can ever exist (a real GLFW limitation, but currently undocumented). Use `glfwSetWindowUserPointer` to attach an `Application*`/`InputState*` and dispatch through that — it's cleaner and removes the global state.

---

## 2. Modern C++ Language Features Not Used

The codebase is built with `CXX_STANDARD 20` (`src/CMakeLists.txt:76`) but reads like C++14 with auto sprinkled in.

### 🟡 2.1 No `[[nodiscard]]` anywhere

Every getter (`getPositionBuffer`, `getParticleCount`, `getViewMatrix`, `compute`, …) and every "did this open?" predicate (`Exporter::isOpen`, `ScreenshotCapture::done`) should be `[[nodiscard]]`. Same for `parseArgs`, factory functions like `initializeWindow`, and `wgpu_utils::initializeWebGPU`. Apply broadly.

### 🟡 2.2 No `std::span` for buffer-like parameters

Functions take `const std::vector<glm::vec4>&` (e.g. `Diagnostics::compute` at `diagnostics.hpp:22`, `Simulation::readbackState` at `simulation.hpp:16`). If a caller has data in an array or a different container, they have to copy. Replace with `std::span<const glm::vec4>` for inputs.

### 🟡 2.3 No `std::string_view` for read-only string parameters

`Exporter::open(const std::string &path, …)` (`exporter.hpp:23`), `Simulation::scenarioName`/`forceMethodName`'s consumers, and `setScenarioName(const char *)` (`gui.hpp:48`) should all take `std::string_view` for cheap, allocation-free passing. Note `scenarioName` returning `const char*` is fine for compile-time strings, but the GUI storing them as `const char*` (`gui.hpp:78`) means a refactor that allocates the string would dangle.

### 🟡 2.4 No `std::optional<T>` for "may return nothing" cases

- `Config parseArgs(int, char**)` returns a `Config` even when argument parsing failed; failures are reported via spdlog warnings and silently ignored. Returning `std::optional<Config>` (or better, `std::expected<Config, std::string>`) makes the failure path explicit.
- `getNextSurfaceViewData` returns `{texture, nullptr}` to signal "no view"; `std::optional<std::pair<…>>` is clearer.
- `mapBufferSync` returns `nullptr` on failure (`wgpu_utils.hpp:56`); same idea.

### 🟡 2.5 `std::tuple` where a struct would be clearer

`initializeWebGPU()` returns `std::tuple<WGPUInstance, WGPUDevice, WGPUAdapter>` (`wgpu_utils.hpp:19`), forcing every caller to use `auto [a, b, c] = …`. A small `struct WebGPUContext { WGPUInstance instance; WGPUDevice device; WGPUAdapter adapter; }` is self-documenting and stable.

Same for `createAndConfigureSurface` returning `std::tuple<WGPUSurface, WGPUTextureFormat>`.

### 🟡 2.6 `enum class` switches without `[[likely]]`/exhaustiveness check

`scenarioName` and `forceMethodName` (`config.cpp:7-22`) end with `return "Unknown";`. With an `enum class` and `-Wswitch-enum`, you'd catch missed cases at compile time. Replace the trailing return with `std::unreachable()` (C++23) or `__builtin_unreachable()` so the compiler flags forgotten enum values.

### 🟡 2.7 No `constexpr`/`consteval` for compile-time data

`PI` is a `static constexpr float` (good — `simulation.cpp:8`), but `kPitchLimit` (`camera.cpp:7`) and others scattered around the codebase are local. Pull common constants into a `constants.hpp` namespaced header.

### 🟡 2.8 `auto` is barely used

`auto` would clean up:

- `auto encoder = createCommandEncoder(device);` — same readability, less typing.
- `auto pass = wgpuCommandEncoderBeginComputePass(...);` — the type name is gibberish anyway.

Keep explicit types when they aid readability (matrices, return types of public APIs).

### 🟡 2.9 `using enum` for switch-statement noise reduction

`config.cpp:7-12`:

```cpp
switch (s) {
case Scenario::TwoBody: return "Two-Body Orbit";
case Scenario::PlummerSphere: return "Plummer Sphere";
case Scenario::RotatingDisk: return "Rotating Disk";
}
```

Could use C++20 `using enum Scenario;` to drop the prefix.

### 🟡 2.10 No `std::ranges` / `std::views`

`config.cpp:62-68` parses a comma-separated list manually. `std::views::split` + `std::ranges::transform` is more idiomatic in C++20. Same in `gpu_tree_builder.cpp:947-995` for the radix params construction.

### 🟢 2.11 Designated initializers

You construct `WGPUBindGroupLayoutEntry`, `WGPURenderPipelineDescriptor`, etc. by zero-init then field-by-field assignment (e.g. `simulation.cpp:414-426`). C++20 designated initializers (`{ .binding = 0, .visibility = WGPUShaderStage_Compute, … }`) are far more readable for descriptor structs.

### 🟢 2.12 No `std::format`

CSV output (`exporter.cpp:22-36`) uses `<<` chains with `std::fixed` and `std::setprecision`. `std::format` (or `fmt::format` since spdlog already pulls fmt in) would be both shorter and more efficient.

---

## 3. Code Duplication

### 🔴 3.1 `Params` struct copy-pasted four times

The 32-byte uniform layout

```cpp
struct alignas(16) Params {
    float dt; float softening; float theta;
    uint32_t numParticles; uint32_t nodeCount; uint32_t paddedN;
    float _pad[2];
};
```

appears literally identically at:
- `simulation.cpp:340-344` (`computeInitialForces`, direct path)
- `simulation.cpp:373-377` (`computeInitialForces`, tree path)
- `simulation.cpp:882-890` (`stepWithDirectForce`)
- `simulation.cpp:1044-1052` (`stepLeapfrogGpuTree`)

Plus the same struct is declared (in WGSL) in five different shader sources. Drift between C++ and WGSL has caused subtle bugs in similar codebases.

**Fix:** define `struct alignas(16) SimParamsUniform { … };` once in `simulation.hpp` (or a new `simulation_params.hpp`). The WGSL definitions can stay verbatim; pin the layout with a single `static_assert(sizeof(SimParamsUniform) == 32);`.

### 🔴 3.2 Bind-group descriptor boilerplate everywhere

The pattern

```cpp
WGPUBindGroupEntry e[N] = {};
e[0].binding = 0;
e[0].buffer = …;
e[0].size = …;
…
WGPUBindGroupDescriptor desc{};
desc.layout = layout;
desc.entryCount = N;
desc.entries = e;
WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &desc);
```

appears **at least 16 times** across `simulation.cpp` (`computeInitialForces`, `stepWithDirectForce`, `ensureBindGroupsCached`), `gpu_tree_builder.cpp` (every cached BG), and `particle_renderer.cpp`. Two helpers would collapse it dramatically:

```cpp
struct BufferBinding { uint32_t slot; WGPUBuffer buffer; uint64_t size; uint64_t offset = 0; };
WGPUBindGroup makeBindGroup(WGPUDevice, WGPUBindGroupLayout,
                            std::initializer_list<BufferBinding>);
```

`gpu_tree_builder.cpp:1080-1086` already has a local `entry()` lambda — promote it.

### 🔴 3.3 Tree-build pass recording duplicated

`recordTreeBuild` (`gpu_tree_builder.cpp:1229-1392`) and `recordTreeBuildTimed` (`gpu_tree_builder.cpp:1398-1601`) are 95% identical. The timed version just splits each phase into its own command encoder + queue submit. A small helper

```cpp
auto runPass = [](WGPUCommandEncoder enc, WGPUComputePipeline pipe,
                  WGPUBindGroup bg, uint32_t wg, uint32_t dynOffset = 0) {
    auto* p = wgpuCommandEncoderBeginComputePass(enc, &passDesc);
    wgpuComputePassEncoderSetPipeline(p, pipe);
    wgpuComputePassEncoderSetBindGroup(p, 0, bg, dynOffset ? 1 : 0,
                                       dynOffset ? &dynOffset : nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(p, wg, 1, 1);
    wgpuComputePassEncoderEnd(p);
    wgpuComputePassEncoderRelease(p);
};
```

plus a single "phase descriptor" list would let both record paths reuse the same body.

### 🟠 3.4 Identical pipeline-creation boilerplate per shader

Each compute shader in `simulation.cpp:410-588` has 30 lines of layout-→-pipeline-→-shader-release plumbing. `gpu_tree_builder.cpp:761-941` already factors this into `storageEntry`/`makeLayout`/`makePipeline` lambdas (good). Lift those lambdas to `wgpu_utils` as functions and use them everywhere.

### 🟠 3.5 Two near-identical t=0 screenshot blocks in `main.cpp`

`main.cpp:122-137` (interactive) and `main.cpp:393-400` (headless). Same logic, slightly different camera matrix. Pull the screenshot loop into a helper.

### 🟡 3.6 Repeated workgroup-size arithmetic

`(N + 255) / 256`, `(N + 127) / 128`, `(N + 63) / 64`, `(N + 1023) / 1024`, `(N + 511) / 512` — all ad-hoc. A single helper:

```cpp
constexpr uint32_t divUp(uint32_t n, uint32_t d) { return (n + d - 1) / d; }
```

with named constants `kForceWG`, `kIntegrateWG`, `kSortBlock`, etc. would make every dispatch self-documenting and fix any future "I changed the workgroup size in the shader and forgot the dispatch" bugs.

---

## 4. Architecture & Design

### 🟠 4.1 `Simulation` is doing four jobs

Currently `Simulation` (`simulation.hpp:10-93`) handles:
1. CPU-side particle initialisation (3 scenarios)
2. GPU buffer ownership
3. Compute pipeline creation / caching
4. Step orchestration (KDK leapfrog) for both direct and tree paths
5. Diagnostics readback support
6. Debug tree dumping

That's a lot of single-responsibility violations. Suggested split:

- `ParticleInitializer` (free functions or a small class) — produces `cpuPositions/Velocities/Colors` from a `Config`. Pure CPU, easily testable.
- `Integrator` — owns kick/drift pipelines, exposes `kick(...)`, `drift(...)`.
- `ForceSolver` (interface) with `DirectForceSolver` and `TreeForceSolver` implementations — owns its own pipeline, layout, and bind group caching. The current `forceMethod_` switch becomes virtual dispatch, which is fine because it happens once per step.
- `Simulation` becomes a coordinator that holds `Integrator + ForceSolver` and runs the leapfrog cycle.

The same idea for `ParticleRenderer` — the trail / fade / reproject code is essentially a separate pass and could live in `TrailEffect`.

### 🟠 4.2 `SimParams` and `Config` overlap

`Config` (`config.hpp:10-24`) has `dt`, `softening`, `theta`, `numParticles`. `SimParams` (`gui.hpp:6-15`) has the same four plus UI flags. `main.cpp:78-82` synchronises them by hand:

```cpp
params.dt = config_.dt;
params.softening = config_.softening;
params.theta = config_.theta;
params.numParticles = config_.numParticles;
```

This will drift. Refactor to one source of truth — e.g. nest a `PhysicsParams` struct that both `Config` and `SimParams` embed:

```cpp
struct PhysicsParams { float dt, softening, theta; int numParticles; };
struct SimParams { PhysicsParams physics; bool paused; bool stepOnce; ...; };
struct Config { PhysicsParams physics; std::string exportPath; ...; };
```

### 🟠 4.3 `Application` is tightly coupled to GLFW and main loop

`Application::loop()` (`main.cpp:145-321`) mixes:
- Frame-time accounting
- Input polling
- Particle-count change handling (recreates simulation!)
- Step orchestration
- Diagnostics readback
- Screenshot capture
- FPS counter
- GUI rendering
- Surface presentation
- Step-limit termination
- Emscripten-specific queue flushing

Split into `update(dt)` and `render(dt)`, and pull screenshot capture / step-limit logic into helpers.

### 🟠 4.4 Hidden lifecycle ordering: "you must call X before Y"

Several APIs have order-dependent contracts that the type system doesn't enforce:

- `Simulation::setSyncTiming` / `setBenchmarkPasses` must be called *between* `initialize()` and the first `step()` (`main.cpp:73-74`).
- `Exporter::writeRow` does nothing unless `open()` was called first; the silent no-op makes typos dangerous.
- `ScreenshotCapture` requires `setTargets` then `initialize` — the order is enforced by inspection of `screenshotCapture_.shouldCapture` returning false.
- `paramsBuffer_.upload(...)` must precede `prepareUploads` must precede `recordTreeBuild` (`simulation.cpp:1060-1067`). A comment ("Upload tree builder data BEFORE creating the command encoder") guards a real WebGPU constraint that nothing in the type system enforces.

**Fix:** make construction encode preconditions. E.g. `Exporter::open` returns an `std::optional<OpenExporter>` that owns the file handle; `writeRow` is a member of `OpenExporter` so you literally cannot call it on a closed exporter.

### 🟠 4.5 `Simulation::reinitialize` is a clone of `initialize` with `clear` first

`simulation.cpp:638-695` duplicates almost everything in `initialize` (`594-636`). Either implement `initialize` as `reinitialize` plus pipeline creation, or unify both into a single `setParticleCount(int)` that handles first-time and resize cases.

### 🟡 4.6 Class members declared in suboptimal order

Several classes order members in a way that allows uninitialised use:

- `Simulation::numParticles_` declared after the buffers (`simulation.hpp:49-67`) but used during construction of buffers in `initialize`. Today initialize sets `numParticles_` first, but if a future helper called `getParticleCount()` before init it would read garbage (default-init to 0 saves us, but only because the header has `= 0`).
- `Application::surfaceFormat_` is uninitialised on declaration (`main.cpp:32`) — relies on `initialize()` setting it before any `loop()` call. This isn't a bug today but is fragile.

Rule of thumb: every member should have an in-class default initialiser.

### 🟡 4.7 `ParticleRenderer::cleanup()` is not idempotent

If called twice, the second call no-ops correctly because every release sets the handle to `nullptr` (`particle_renderer.cpp:652-715`). Good. But it is explicitly invoked from `~Application`, **and** the destructor of `wgpu_utils::Buffer` runs again on members. `ParticleRenderer` has no destructor at all — meaning if you forget to call `cleanup()`, you leak everything. Add `~ParticleRenderer() { cleanup(); }`.

Same applies to `ScreenshotCapture` (`screenshot.hpp:7-36`).

### 🟡 4.8 Friend-style coupling: `ScreenshotCapture::capture` knows too much

`screenshot.hpp:14-19`:

```cpp
void capture(WGPUDevice, WGPUQueue, ParticleRenderer&,
             WGPUBuffer posBuffer, WGPUBuffer colorBuffer,
             int particleCount, ...);
```

The simulation already exposes positions/colors via getters; `capture` accepts them as raw handles. Cleaner: pass a `const Simulation&` (or a `ParticleData` struct), and let `ScreenshotCapture` ask for what it needs.

---

## 5. Const-Correctness & API Hygiene

### 🟠 5.1 Many getters not `const` despite being trivial

- `Simulation::getLastTiming()` (`simulation.hpp:29`) and similar: already const, fine.
- `Buffer::get()` (`wgpu_utils.hpp:40`) is `const` but returns a non-const handle that can be mutated — fundamentally OK for WebGPU semantics.
- `GUI::getParams()` returns mutable reference (`gui.hpp:28`) — by design for ImGui sliders. Fine, but document it.

Less obvious:
- `Camera::getPosition` (`camera.hpp:15`) is `const`. Good. But `getForward`/`getRight` are private and would benefit from `[[nodiscard]] const`.
- `DiagnosticsCalculator::reset()` (`diagnostics.hpp:25`) modifies state. Not const. Fine.

### 🟠 5.2 Public mutable callbacks via globals

§1.9 covers this — once you remove the file-statics in `graphics.cpp`, the surface gets cleaner.

### 🟡 5.3 `int` everywhere where `size_t` or `uint32_t` is correct

`Simulation::getParticleCount()` returns `int` (`simulation.hpp:22`) but is immediately cast to `uint32_t` for WebGPU's `numParticles` field everywhere it's used. `numParticles_` is stored as `int` (`simulation.hpp:49`). The CLI parser reads it as `std::stoi` (could be negative). The constraint `0 < N <= 100000` is enforced only by the GUI clamp at `gui.cpp:89-90`.

Pick one type, ideally `uint32_t` (matches the WGSL uniform), and validate at the boundary. This will eliminate the `static_cast<uint32_t>(numParticles_)` litter throughout `simulation.cpp` and `gpu_tree_builder.cpp`.

### 🟡 5.4 `void *begin` is a confusing parameter name

`Buffer::upload(void *begin, size_t bytes, …)` (`wgpu_utils.hpp:45`). `begin` suggests a beginning iterator; `data` or `src` would be clearer. Better still:

```cpp
void upload(WGPUQueue queue, std::span<const std::byte> data, bool wait = false);
```

so you can't pass mismatched pointer/size pairs.

---

## 6. Type Safety & Numeric Conversions

### 🟠 6.1 Implicit narrowing in many places

- `simulation.cpp:9` `int N = static_cast<int>(positions.size());` — fine.
- `simulation.cpp:894` `params.numParticles = static_cast<uint32_t>(numParticles_);` — fine but should not be needed (see §5.3).
- `screenshot.cpp:9` `int N = static_cast<int>(positions.size());` — same.
- `diagnostics.cpp:9-10` mixes `int N`, `double softSq`, `static_cast<double>(softening)`. The implicit `float→double` promotion in `softening * softening` is fine in C++ but `-Wconversion` will scream.

Consider turning on `-Wconversion -Wsign-conversion -Wshorten-64-to-32` and fix the warnings methodically.

### 🟠 6.2 Mixed `uint32_t` / `size_t` in `ScreenshotCapture::nextTarget_`

`screenshot.hpp:34` declares `size_t nextTarget_`. `screenshot.cpp:21` `setTargets(...)` resets it to 0. Comparisons at `screenshot.cpp:48` against `targets_.size()` are fine. But `done()` (`screenshot.hpp:21`) returns `bool` from `nextTarget_ >= targets_.size()` — perfect. Just be aware that any future arithmetic on `nextTarget_` could underflow.

### 🟡 6.3 Float comparisons without epsilon

`camera.cpp:52` `if (lastMouseX_ == 0.0 && lastMouseY_ == 0.0)` — used as a sentinel for "first move", but a real mouse could legitimately be at origin pixel (0,0). Use `std::optional<glm::dvec2>` to express "no previous position".

### 🟡 6.4 `static_cast<int>(width_)` when `uint32_t→int` could overflow

`screenshot.cpp:131-133`. Width/height come from a CLI / config and will not be huge, but the cast is potentially lossy. Use `gsl::narrow` or `std::numeric_limits` checks at the boundary, or just keep them as `uint32_t` everywhere and cast at the C-API call site.

---

## 7. Error Handling

### 🟠 7.1 Inconsistent strategy: throw vs log-and-continue vs silently fail

- `wgpu_utils::createShaderModule` throws on failure (`wgpu_utils.cpp:331-333`).
- `wgpu_utils::createCommandEncoder` throws (`wgpu_utils.cpp:346-348`).
- `Simulation::createComputePipelines` logs `error` and continues with `pipeline_ = nullptr` (`simulation.cpp:447-450`, etc.). The next `step()` will then `wgpuComputePassEncoderSetPipeline(pass, nullptr)` — undefined behaviour at the WebGPU layer.
- `parseArgs` warns and uses defaults (`config.cpp:78-80`).
- `Diagnostics::compute` silently skips potential energy when `N > 5000` (`diagnostics.cpp:22`).

Pick a discipline:
1. Constructors / `initialize` may throw; the whole app exits on init failure.
2. Hot-loop functions must not throw; they should return `bool` or `std::expected`.
3. Optional features (potential energy, screenshots) should be visibly opt-in with explicit logging when skipped.

### 🟠 7.2 WebGPU error callbacks are global-singleton

`wgpu_utils.cpp:112-116` registers an uncaptured error callback that just logs. Multiple devices would all share a single log line — fine for now (you only ever create one device), but the device-lost callback (`wgpu_utils.cpp:136-140`) similarly logs and does **not** trigger any cleanup or reinitialisation. A real device loss (e.g. driver reset) leaves the application in an unrecoverable state with no signal to the user.

### 🟡 7.3 `mapBufferSync` busy-loops indefinitely

`wgpu_utils.cpp:384-387`:

```cpp
while (!context.done) {
    wgpuPollEvents(device, true);
}
```

If the map-async never resolves (driver bug, lost device), the program hangs forever. Add a timeout / max-iteration guard.

### 🟡 7.4 No graceful handling of `screenshotTimes` past `maxSteps`

`main.cpp:194-208` will simply never trigger captures if the targets are after the simulation ends; user gets silent partial output. Warn at startup if any target is unreachable.

---

## 8. Logging Discipline

### 🔴 8.1 `spdlog::info` in hot paths

This will visibly slow down headless benchmarks and pollute the console.

- `simulation.cpp:336` `spdlog::info("[INITFORCE] N={} forceMethod={}", …)` — once at init, fine.
- `simulation.cpp:379-402` — inside a code path called every reinitialisation; `info` not `debug`.
- `simulation.cpp:640-694` — every reinit logs ~13 info lines.
- `simulation.cpp:705-768` (`ensureBindGroupsCached`) — logs **on every bind-group cache hit**. So every step prints "[SIM-BG] Cache hit, N=..."! See `simulation.cpp:706`.
- `simulation.cpp:862-864` — `spdlog::debug` (good) but with multi-arg formatting.
- `simulation.cpp:186-189` — `spdlog::info("[MAIN] Calling step(), …")` **every step**.
- `gpu_tree_builder.cpp:1031-1032` — same pattern, hits every frame.

For 5000 steps × 5 messages/step you're paying real I/O. Tag these `debug`, gate them behind `--verbose`, or remove them.

**Fix:** audit every `spdlog::info` outside of one-shot init/teardown paths, demote to `debug`, and use spdlog's level filtering.

### 🟠 8.2 Tracing `void*` formatting littered through critical code

`simulation.cpp` has 17 occurrences of `(void*)foo.get()` in `spdlog::info(...)`. These are debug breadcrumbs. Once you trust the bind-group caching, delete them. If they're load-bearing for debugging, gate behind `#ifdef GALAXYSIM_DEBUG_GPU` or a log level.

### 🟡 8.3 `spdlog::set_level` initialization in `main` is awkward

`main.cpp:486-496`:

```cpp
for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--verbose") { ... }
}
if (spdlog::get_level() != spdlog::level::debug) {
    spdlog::set_level(spdlog::level::info);
}
```

Pre-scanning argv twice is fragile. Move this into `parseArgs` and surface a `Config::verbose` field.

---

## 9. Magic Numbers & Hidden Constants

Many physical and tuning constants are hard-coded in different places:

- `simulation.cpp:8` `PI = 3.14159265358979323846f` — duplicate of `M_PI` (used in `diagnostics.cpp:81`); inconsistent. Use `std::numbers::pi_v<float>` (C++20) — `<numbers>`.
- `simulation.cpp:206` `mass = 1000.0f`, `separation = 10.0f` (two-body scenario) — should be configurable or named constants.
- `simulation.cpp:235` `a = 5.0f` (Plummer scale length) — same.
- `simulation.cpp:243` `std::max(0.001f, std::min(u, 0.999f))` — clamps that hide divide-by-zero protection.
- `simulation.cpp:303` `r = std::min(r, 50.0f);` — disk cutoff.
- `simulation.cpp:315` `0.5f` rotation factor — unexplained.
- `gui.cpp:38-43` style constants and `1.5f` font scale, `100000` particle cap (`gui.cpp:90`).
- `main.cpp:88-89` `camera_.setDistance(80.0f)` and `RotatingDisk` default — already in config but not centralised.
- `gpu_tree_builder.cpp:1093-1094` `numParticles * 16` (sizeof vec4), `numWorkgroups * 2 * 16` etc. — all bare arithmetic.
- `simulation.cpp:728-731` `e[0].size = 32` — hard-coded uniform size, must match `sizeof(SimParamsUniform)`.

**Fix:** introduce a `constants.hpp` (or `physics_constants.hpp` and `gpu_constants.hpp`) with `inline constexpr` values. Replace hard-coded `32`, `16`, `64`, `128`, `256`, `1024` with named constants pinned by `static_assert(sizeof(...) == ...)`.

---

## 10. Header Hygiene

### 🟡 10.1 Includes pulled into headers unnecessarily

- `screenshot.hpp:3` includes `particle_renderer.hpp`. The only use is in `capture(... ParticleRenderer &renderer ...)`. A forward declaration is enough.
- `simulation.hpp:4-7` includes `config.hpp`, `exporter.hpp`, `gpu_tree_builder.hpp`, `wgpu_utils.hpp`. The first three are for *struct* members like `StepTiming` and `GpuTreeBuilder`. Those are concrete (not pointers), so you do need the headers — but `Config` is only used as a `const Config&` parameter; forward-declare it.
- `gui.hpp:3` pulls in `graphics.hpp`, which pulls in `<GLFW/glfw3.h>`, `glfw3webgpu.h`, etc. into every TU that uses GUI.

### 🟡 10.2 `using namespace` at top of `main.cpp`

`main.cpp:20-21`:

```cpp
using namespace wgpu_utils;
using namespace graphics;
```

Pulls everything into the global namespace of the TU. Fine for a small executable but inconsistent (`gui.cpp:7` does the same, others don't). Drop them and qualify, or use namespace aliases (`namespace wu = wgpu_utils;`).

### 🟡 10.3 Missing `#pragma once` consistency

All headers use `#pragma once`. Good. Consider also adding traditional include guards for portability (cmake already targets GCC/Clang/MSVC where `#pragma once` is universally supported, so this is low priority).

### 🟢 10.4 Implementation details exposed in headers

`wgpu_utils.hpp:28-47` exposes private members (`m_buffer`, `m_device`, …) — that's fine, they're truly private. But `Simulation::cpuPositions_` (`simulation.hpp:59-61`) is held in the simulation only because `initParticles` writes to it before upload. After `initialize()` it's wasted memory (per particle: 3×16 = 48 bytes × 100k = 4.8 MB). Free it after upload.

---

## 11. Build System & Tooling

### 🟠 11.1 `COMPILE_WARNING_AS_ERROR OFF` and minimal warning flags

`src/CMakeLists.txt:79`:

```cmake
COMPILE_WARNING_AS_ERROR OFF
```

Combined with the absence of `-Wall -Wextra -Wpedantic -Wconversion` flags, the build accepts almost any warning. Many of the §6 issues would be auto-detected with `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2`.

For Emscripten, `target_compile_options(imgui_lib PRIVATE -Wno-error)` (`src/CMakeLists.txt:14`) suggests warnings *were* being errored at some point. Re-enable for first-party libs only (`galaxysim` and your own `*_lib` targets).

### 🟠 11.2 Each component is its own static library — fine, but unused

The CMakeLists builds `wgpu_utils`, `graphics_lib`, `camera_lib`, …, eight static libs total, but only the executable links them. Static libs only buy you incremental builds, not module isolation, and they don't compose well with header-only template usage. Either:
1. Embrace it: add unit tests per library, or
2. Collapse to a single `galaxysim_core` library (or even just compile `main.cpp` against globbed sources). Faster CMake processing.

A C++20 module split (one module per current library) would be the *modern* answer but requires CMake 3.28+ and toolchain changes. Probably not worth it here.

### 🟡 11.3 No clang-format / clang-tidy config

There are subtle style mixes: 4-space indent (good, consistent), but braces vary between K&R and Allman in different functions; sometimes `WGPU…Descriptor desc = {};` other times `desc{};`. A `.clang-format` and a `.clang-tidy` (with at least `bugprone-*`, `cppcoreguidelines-*`, `modernize-*`, `performance-*`, `readability-*` checks) would catch many issues here automatically.

### 🟡 11.4 No `target_compile_features`

`src/CMakeLists.txt:76-79` sets `CXX_STANDARD 20` only on the executable. The libraries use whatever the default is (likely C++17 or whatever was inherited via `target_link_libraries`). Use `target_compile_features(galaxysim_core PUBLIC cxx_std_20)` on each library so they propagate the requirement.

### 🟡 11.5 No sanitizer build option

Add a `GALAXYSIM_SANITIZE=address|undefined|thread` cache option that adds `-fsanitize=...` to all `*_lib` and `galaxysim`. Useful for the inevitable "I have a memory error somewhere" debugging.

### 🟡 11.6 Hardcoded `100000u` particle limit in `simulation.cpp:628`

```cpp
gpuTreeBuilder_.initialize(device, 100000u);
```

Same magic number is in `gui.cpp:90`. Pull into `Config` (or `constants.hpp`) so they can never drift.

### 🟢 11.7 `target_copy_webgpu_binaries` only for the executable

Not an issue today but if you ever add a test executable you'll need this for it too. Wrap in a function or apply to a list.

### 🟢 11.8 Emscripten linker options not version-pinned

`src/CMakeLists.txt:101-106` sets `-sASYNCIFY`, etc. without a version comment. Document the emsdk version that was tested.

---

## 12. Testing

### 🔴 12.1 No tests at all

There is no `tests/` directory, no GoogleTest / Catch2 / doctest dependency, no `enable_testing()` call. Several modules are perfectly testable without a GPU:

- `parseArgs` — pure CPU function, easy to wrap in unit tests.
- `Diagnostics::compute` — given a fixed input vector, asserts on conservation values.
- Camera math — `getForward`, `getRight`, view matrix.
- `getPaddedN` — bit-twiddling, easy to property-test.
- Two-body Kepler validation: in headless mode you already export `--export results.csv`; a Python or C++ test could run a fixed config and assert energy drift < threshold.

**Fix (initial scope, ~1 day):** add `doctest` (single-header) as a dependency, create `tests/test_diagnostics.cpp`, `tests/test_config.cpp`, `tests/test_camera.cpp`, `tests/test_padded_n.cpp`. Wire up `enable_testing()` and `add_test()` in CMake.

### 🟠 12.2 No integration / regression test for the GPU path

Conservation laws are excellent invariants. A headless smoke test:

```bash
./galaxysim --headless --scenario twobody --steps 100 --export /tmp/twobody.csv
python -c "import csv,sys;rows=list(csv.DictReader(open('/tmp/twobody.csv'))); \
           drift=float(rows[-1]['energy_drift']); sys.exit(0 if drift<1e-3 else 1)"
```

run as a CI check, would catch any regression in physics correctness.

### 🟡 12.3 No CI configuration

The `.github/` directory exists but I didn't see a workflow file in the listing. Add a minimal GitHub Actions workflow that builds the project on Linux/macOS with the WGPU backend, builds for Emscripten, and runs the unit/regression tests above.

---

## 13. WebGPU-Specific Issues

### 🟠 13.1 `WGPUTextureViewDescriptor` left half-uninitialised

`graphics.cpp:19-31` declares `WGPUTextureViewDescriptor viewDescriptor;` (no `{}` zero-init) then sets some fields. The `nextInChain` pointer comes from `<webgpu/webgpu.h>` and is by convention zero-init via `{}`. Some backends will reject undefined `nextInChain` values. Use `WGPUTextureViewDescriptor viewDescriptor{};`.

Similar elsewhere — search for "WGPU.*Descriptor [a-zA-Z]+;" without a `{}`.

### 🟠 13.2 `WGPUSurfaceConfiguration` does not handle resize

`graphics.cpp:34-55` configures the surface once on startup. Window resize (which is currently disabled — `glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE)` at `graphics.cpp:63`) would otherwise crash because the surface isn't reconfigured and the depth texture isn't resized. If you ever enable resizing, you need an `onResize(uint32_t w, uint32_t h)` path.

### 🟡 13.3 Bind-group cache invalidation by particle count only

`Simulation::ensureBindGroupsCached` (`simulation.cpp:704-769`) only checks `cachedBGParticleCount_ == numParticles_`. If a new buffer with the same particle count is bound (e.g. because `reinitialize` ran), the cache happily returns stale handles. `reinitialize` does call `invalidateBindGroups()` first (`simulation.cpp:649`), so today this works. But the `gpu_tree_builder.cpp` cache *also* checks the buffer pointers (`gpu_tree_builder.cpp:1028-1030`), which is more defensive — apply the same check in `Simulation`.

### 🟡 13.4 Same-encoder timing is misleading

`stepLeapfrogGpuTree` (`simulation.cpp:1218-1224`) records timestamps `t0..t4` around CPU command-encoder calls, not GPU work. Without `--sync-timing`, those times are essentially CPU recording cost, not pass cost. The comment says "// Per-pass breakdown (populated when --benchmark-passes)" — make the non-breakdown path's units explicit (e.g. rename `forceMs` to `forceCpuRecMs`) or drop it.

---

## 14. Suggested Refactor Order

If you are doing this manually, doing them roughly in this order minimises rework:

1. **Add `.clang-format` and run it.** Mechanical; one commit.
2. **Add `[[nodiscard]]` and turn on `-Wall -Wextra -Wpedantic -Wconversion`.** Fix the warnings file by file.
3. **Introduce `constants.hpp` and replace magic numbers.** Pull `PI`, `MAX_PARTICLES`, workgroup sizes, sizeof-uniforms into named constants.
4. **Extract `SimParamsUniform`.** Single struct used everywhere; static_assert the layout.
5. **Add WebGPU RAII wrappers (`WgpuUnique`).** Replace the manual release pairs incrementally — one resource type at a time so each commit compiles.
6. **Add destructors / `cleanup()` to `Simulation` and `GpuTreeBuilder`.** Closes the leaks.
7. **Demote / remove hot-path `spdlog::info` calls.** Big perf win for benchmarks.
8. **Deduplicate bind-group creation with `makeBindGroup` helper.** Major LOC reduction.
9. **Split `Simulation` into `Integrator` + `ForceSolver` (interface) + `ParticleInitializer`.** Architectural; do after the cleanups so the interface boundaries are clear.
10. **Unify `Config` and `SimParams` via a shared `PhysicsParams`.**
11. **Add tests.** Start with `parseArgs`, `Diagnostics`, `Camera`, `getPaddedN`. Wire CI.
12. **Add a `WgpuContext` struct to replace `std::tuple` returns** and small refactors that improve self-documentation.

A reasonable expectation: steps 1-8 are 2-3 focused days of work and net you a ~30% LOC reduction with a much safer codebase. Steps 9-12 are bigger commitments and should be planned per-feature.

---

## Appendix: Per-File Quick Notes

| File                       | Lines | Most pressing issue                                         |
|----------------------------|------:|-------------------------------------------------------------|
| `main.cpp`                 |   528 | `Application::loop` is 175 lines; split & member ordering   |
| `simulation.cpp`           |  1226 | Pipeline leak (no dtor); `Params` struct duplicated 4x      |
| `simulation.hpp`           |    93 | No dtor, members declared before count                       |
| `gpu_tree_builder.cpp`     |  1601 | Pipeline leak; `recordTreeBuild` ≈ `recordTreeBuildTimed`   |
| `gpu_tree_builder.hpp`     |   131 | No dtor — leaks 11 pipelines + 11 layouts                   |
| `particle_renderer.cpp`    |   716 | No dtor; trail logic could be its own class                 |
| `particle_renderer.hpp`    |    60 | No dtor                                                     |
| `wgpu_utils.cpp`           |   391 | `Buffer` rule-of-five violation; pointless try/catch        |
| `wgpu_utils.hpp`           |    58 | Missing rule-of-five; tuple returns                         |
| `graphics.cpp`             |   126 | File-statics for callbacks                                  |
| `graphics.hpp`             |    47 | Pulls GLFW into every consumer                              |
| `gui.cpp`                  |   156 | Mostly fine; wgpuQueueRelease in `render()` is unnecessary  |
| `gui.hpp`                  |    80 | `const char*` storage for scenario/method names             |
| `screenshot.cpp`           |   158 | `STB_IMAGE_WRITE_IMPLEMENTATION` exposed via header chain   |
| `screenshot.hpp`           |    36 | No dtor; takes `ParticleRenderer&` by ref                   |
| `camera.cpp`               |    96 | Float `==` sentinel; otherwise clean                        |
| `camera.hpp`               |    43 | Mostly clean                                                |
| `config.cpp`               |    93 | `parseArgs` should return `std::expected<Config, error>`    |
| `config.hpp`               |    29 | Fine as a POD                                               |
| `diagnostics.cpp`          |    91 | `M_PI` — replace with `std::numbers::pi`                    |
| `diagnostics.hpp`          |    34 | Could take `std::span`                                      |
| `exporter.cpp`             |    42 | Could use `std::format` / `fmt`                             |
| `exporter.hpp`             |    33 | Fine                                                        |

---

*End of audit. If you want me to expand any section into concrete code patches, point at a specific item by its number (e.g. "do §1.4") and I'll prepare the change.*
