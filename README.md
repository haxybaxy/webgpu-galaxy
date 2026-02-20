# GalaxySim

Real-time N-body galaxy simulator using the Barnes-Hut algorithm. Runs gravitational physics on both CPU and GPU in parallel using WebGPU compute shaders, with an interactive 3D viewer built on Dear ImGui.

Supports up to 100,000 particles with O(N log N) force computation, symplectic Leapfrog integration, and per-step energy/momentum conservation diagnostics.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![WebGPU](https://img.shields.io/badge/GPU-WebGPU-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Barnes-Hut octree** gravity solver running on the GPU via WGSL compute shaders (O(N log N) per step)
- **CPU-GPU dual-track** architecture: physics computed on both CPU and GPU every step, avoiding expensive GPU readbacks
- **KDK Leapfrog integrator** (symplectic, 2nd-order) with Euler fallback
- **Three initial-condition scenarios**: rotating disk galaxy, Plummer sphere, two-body orbit
- **Real-time GUI** with adjustable timestep, softening, Barnes-Hut opening angle, and particle count
- **Conservation diagnostics**: kinetic energy, potential energy, energy drift, momentum magnitude
- **Per-step timing breakdown**: octree build, force computation, integration
- **Headless batch mode** for automated parameter sweeps with CSV export
- **Cross-platform**: native desktop (macOS, Linux, Windows) and WebAssembly via Emscripten
- **Zero manual dependencies**: everything fetched automatically via CMake FetchContent

## Requirements

- **CMake** 3.15+
- **C++20 compiler** (Clang 14+, GCC 12+, MSVC 2022+)
- **GPU with Vulkan, Metal, or DX12 support** (for WebGPU backend)

For WebAssembly builds:
- **Emscripten SDK** 3.1.50+

All library dependencies are downloaded automatically during configuration:

| Dependency | Version | Purpose |
|------------|---------|---------|
| [WebGPU-distribution](https://github.com/eliemichel/WebGPU-distribution) | 0.2.0 | WebGPU C API (wgpu-native or Dawn) |
| [GLFW](https://github.com/glfw/glfw) | 3.4 | Window and input management |
| [glfw3webgpu](https://github.com/eliemichel/glfw3webgpu) | 1.2.0 | GLFW-WebGPU surface bridge |
| [spdlog](https://github.com/gabime/spdlog) | 1.16.0 | Logging |
| [Dear ImGui](https://github.com/ocornut/imgui) | 1.90.9 | GUI controls and diagnostics |
| [GLM](https://github.com/g-truc/glm) | 1.0.2 | Math (vectors, matrices) |

## Building

### Native (macOS / Linux / Windows)

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run
./build/src/galaxysim
```

### Debug build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### WebGPU backend selection

The default backend is **wgpu-native**. You can switch to Dawn or target Emscripten:

```bash
# wgpu-native (default)
cmake -B build -DWEBGPU_BACKEND=WGPU

# Dawn (Google's WebGPU implementation)
cmake -B build -DWEBGPU_BACKEND=DAWN

# Build Dawn from source instead of using prebuilt binaries
cmake -B build -DWEBGPU_BACKEND=DAWN -DWEBGPU_BUILD_FROM_SOURCE=ON
```

### Emscripten (WebAssembly)

See the detailed [Running in the Browser](#running-in-the-browser) section below.

## Usage

### Interactive mode (default)

```bash
./build/src/galaxysim
```

Opens a 1280x720 window with a rotating disk galaxy of 10,000 particles. The simulation starts paused -- press **Play** in the GUI or use the controls below.

### Command-line options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--scenario` | `disk`, `plummer`, `twobody` | `disk` | Initial particle distribution |
| `--integrator` | `leapfrog`, `euler` | `leapfrog` | Time integration method |
| `--N` | 2 -- 100000 | `10000` | Number of particles |
| `--dt` | float | `0.001` | Timestep |
| `--softening` | float | `0.5` | Gravitational softening length |
| `--theta` | float | `0.75` | Barnes-Hut opening angle (lower = more accurate, slower) |
| `--seed` | uint32 | `42` | Random seed for initial conditions |
| `--steps` | int | `0` (unlimited) | Stop after N steps (0 = run forever in interactive mode) |
| `--export` | filepath | (none) | Write per-step CSV diagnostics to file |
| `--headless` | flag | off | Run without a window (batch mode) |
| `--verbose` | flag | off | Enable debug-level logging |

### Examples

```bash
# Large Plummer sphere with high accuracy
./build/src/galaxysim --scenario plummer --N 50000 --theta 0.5

# Quick two-body orbit test
./build/src/galaxysim --scenario twobody --dt 0.0005

# Headless benchmark: 5000 steps, export results
./build/src/galaxysim --headless --scenario disk --N 20000 --steps 5000 --export results.csv

# Euler integrator comparison
./build/src/galaxysim --integrator euler --N 5000 --steps 2000 --export euler.csv

# Reproducible run with specific seed
./build/src/galaxysim --scenario plummer --seed 12345 --N 10000
```

### Headless batch mode

When `--headless` is specified, the simulator runs without creating a window. This is useful for benchmarks, parameter sweeps, and CI. Requires `--steps` (defaults to 1000 if omitted). Progress is logged every 10% of steps.

```bash
./build/src/galaxysim --headless --scenario plummer --N 10000 --steps 5000 --export run.csv
```

### Interactive controls

**Camera:**

| Input | Action |
|-------|--------|
| Left-click drag | Orbit (rotate view) |
| Scroll wheel | Zoom in/out |
| W / S | Move camera forward / backward |
| A / D | Move camera left / right |
| Space / Shift | Move camera up / down |

**GUI panel (Dear ImGui):**

- **Play / Pause** -- toggle simulation stepping
- **Step** -- advance a single step while paused
- **Timestep** slider -- adjust dt (0.0001 to 0.01)
- **Softening** slider -- gravitational softening (0.01 to 5.0)
- **Theta (BH)** slider -- Barnes-Hut opening angle (0.1 to 2.0)
- **Particle Count** -- change N and reinitialize (2 to 100,000)
- **Diagnostics** panel -- kinetic energy, potential energy, total energy, energy drift, momentum magnitude
- **Timing** panel -- per-step breakdown of octree build, force computation, and integration in milliseconds

## Scenarios

### Rotating Disk (`--scenario disk`)

Exponential disk galaxy with vertical thickness that decreases with radius. Particles are initialized with circular orbital velocities based on enclosed mass. Masses are drawn uniformly from [0.5, 2.0]. Particles are colored from blue (center) to red (edge).

### Plummer Sphere (`--scenario plummer`)

Spherically symmetric star cluster following the Plummer density profile (Aarseth et al. 1974). Radii are sampled from the Plummer CDF, and velocities are assigned via rejection sampling of the Plummer distribution function. All particles have unit mass. Colors fade from white (inner) to blue (outer).

### Two-Body Orbit (`--scenario twobody`)

Two equal-mass bodies (m=1000) in a circular orbit. Always uses exactly 2 particles regardless of `--N`. Useful for validating energy conservation and integrator accuracy. The orbit velocity accounts for gravitational softening.

## CSV Export Format

When `--export <path>` is specified, a CSV file is written with one row per simulation step:

```
step,time,kinetic_energy,potential_energy,total_energy,energy_drift,px,py,pz,tree_build_ms,force_ms,integrate_ms
```

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Step number (1-indexed) |
| `time` | float | Simulation time (cumulative dt) |
| `kinetic_energy` | float | Total kinetic energy: sum of 0.5 * m * v^2 |
| `potential_energy` | float | Gravitational potential energy (0 if N > 5000) |
| `total_energy` | float | KE + PE |
| `energy_drift` | float | \|E(t) - E(0)\| / \|E(0)\| |
| `px`, `py`, `pz` | float | Total momentum components |
| `tree_build_ms` | float | Octree construction time (ms) |
| `force_ms` | float | Force computation time (ms) |
| `integrate_ms` | float | Integration time (ms) |

Note: potential energy uses an O(N^2) direct sum and is only computed when N <= 5000. For larger particle counts, `potential_energy` and the PE contribution to `total_energy` and `energy_drift` will be zero.

## Architecture

### CPU-GPU Hybrid Dual-Track

Every simulation step runs equivalent physics on both CPU and GPU in parallel. The CPU maintains mirror arrays (`cpuPositions_`, `cpuVelocities_`, `cpuAccelerations_`) that track GPU buffer state. This avoids expensive GPU-to-CPU readbacks -- the CPU mirror is used for octree construction (which requires random-access spatial queries) and for computing conservation diagnostics.

### Simulation Step (KDK Leapfrog)

The default integrator is the Kick-Drift-Kick (KDK) Leapfrog scheme, a symplectic 2nd-order method:

1. **Half-kick**: `v += a * dt/2` (CPU loop + GPU kick shader)
2. **Drift**: `x += v * dt` (CPU loop + GPU drift shader)
3. **Octree build**: `Octree::build()` from CPU positions, flatten to `OctreeNode[]`, upload to GPU
4. **Force computation**: Barnes-Hut traversal on both CPU (loop over particles) and GPU (compute shader with explicit stack, depth 64)
5. **Second half-kick**: `v += a * dt/2` (CPU loop + GPU kick shader)

The Euler integrator (`--integrator euler`) follows a simpler force-then-integrate sequence and is preserved as a first-order fallback for comparison.

### Barnes-Hut Algorithm

The octree is built on the CPU each step from the mirror position array. Each internal node stores the center of mass and total mass of all particles in its subtree. The force shader traverses the tree with the opening criterion:

```
halfWidth^2 / distSq < theta^2
```

When this condition is met, the node is treated as a single body at its center of mass. Lower theta values force deeper traversal (more accurate, slower). The recommended range is 0.3--1.0.

### OctreeNode GPU Layout

```cpp
struct OctreeNode {
    glm::vec4 centerOfMass;  // xyz = center of mass, w = total mass
    glm::vec4 bounds;        // xyz = bounding box center, w = half-width
    int32_t children[8];     // child node indices, -1 = empty
};
```

### Shader Pipeline

All shaders are WGSL compute shaders embedded as C string literals (no separate `.wgsl` files):

| Shader | Workgroup Size | Purpose |
|--------|---------------|---------|
| Force | 64 | Barnes-Hut tree traversal, writes accelerations |
| Kick | 256 | Half-step velocity update: `v += a * dt/2` |
| Drift | 256 | Position update: `x += v * dt` |
| Integrate (Euler) | 256 | Combined velocity + position update (Euler fallback) |

### Rendering

Particles are rendered as instanced billboard quads (6 vertices per quad, 2 triangles) with **additive blending**. Positions and colors are read directly from GPU storage buffers using `@builtin(instance_index)` -- no vertex buffer copies. A depth buffer is used for z-ordering.

### Params Uniform Buffer (32 bytes)

```
offset 0:  f32 dt
offset 4:  f32 softening
offset 8:  f32 theta
offset 12: u32 numParticles
offset 16: u32 nodeCount
offset 20: f32[3] padding
```

This struct is `alignas(16)` in C++ and must match the WGSL layout exactly.

## Project Structure

```
galaxysim/
  CMakeLists.txt          Root CMake configuration
  CLAUDE.md               AI assistant context
  README.md               This file
  imgui.ini               Dear ImGui window layout state
  src/
    CMakeLists.txt         Source build rules (per-file static libraries)
    main.cpp               Application class (interactive) + runHeadless() (batch)
    simulation.cpp/.hpp    Physics engine: embedded shaders, scenarios, integrators, CPU-GPU step
    octree.cpp/.hpp        Octree: recursive insert, center-of-mass propagation, flatten to GPU
    particle_renderer.cpp/.hpp  WebGPU render pipeline with embedded vertex/fragment shader
    camera.cpp/.hpp        Orbital camera with WASD + mouse controls
    gui.cpp/.hpp           Dear ImGui control panel and diagnostics display
    diagnostics.cpp/.hpp   Energy and momentum conservation metrics
    config.cpp/.hpp        CLI argument parser and Config struct
    exporter.cpp/.hpp      Per-step CSV export
    graphics.cpp/.hpp      GLFW/WebGPU window and surface abstraction
    wgpu_utils.cpp/.hpp    WebGPU helpers (device init, buffer management, shader compilation)
```

Each `.cpp` file is compiled as its own static library and linked into the final `galaxysim` executable.

## Physics Parameters Guide

### Timestep (`--dt`)

Controls simulation accuracy and stability. Smaller values are more accurate but slower per unit of simulation time.

- **0.0001**: High precision, good for two-body orbit validation
- **0.001**: Default, good balance for most scenarios
- **0.005--0.01**: Fast but may show energy drift in long runs

### Softening (`--softening`)

Prevents singularities in gravitational force at close range. The force law becomes: `F = G * m1 * m2 * r / (r^2 + eps^2)^(3/2)`.

- **0.01--0.1**: Near-Newtonian, may cause close encounters
- **0.5**: Default, smooth behavior for disk and Plummer scenarios
- **1.0--5.0**: Very soft, suppresses small-scale structure

### Opening Angle (`--theta`)

Controls the accuracy-performance tradeoff of the Barnes-Hut algorithm. Nodes subtending an angle smaller than theta are approximated as point masses.

- **0.3**: High accuracy, slow (many node openings)
- **0.5**: Good accuracy for quantitative work
- **0.75**: Default, good visual quality with reasonable performance
- **1.0--2.0**: Fast but increasingly approximate

### Particle Count (`--N`)

- **2**: Two-body (forced by `--scenario twobody`)
- **1,000--5,000**: Potential energy computed, full conservation diagnostics
- **10,000**: Default, good interactive performance
- **50,000--100,000**: Requires capable GPU, octree build becomes the bottleneck

## Running in the Browser

GalaxySim compiles to WebAssembly and runs in any browser with WebGPU support. The build produces a self-contained `.html` page with the simulation canvas and the full Dear ImGui interface.

### Prerequisites

You need the **Emscripten SDK** (emsdk). A local copy is included in the repo under `emsdk/`, or you can install your own:

```bash
# Option A: Use the bundled emsdk
cd emsdk
./emsdk install latest
./emsdk activate latest
cd ..

# Option B: Install globally (https://emscripten.org/docs/getting_started)
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest && cd ..
```

Minimum version: **Emscripten 3.1.50+** (required for WebGPU support).

### Building the WASM bundle

```bash
# 1. Activate emsdk in your current shell
source emsdk/emsdk_env.sh

# 2. Configure with emcmake (uses Emscripten's WebGPU backend automatically)
emcmake cmake -B build-web

# 3. Build
cmake --build build-web
```

This produces three files in `build-web/src/`:

| File | Description |
|------|-------------|
| `galaxysim.html` | Entry page with Emscripten shell (canvas + console) |
| `galaxysim.js` | JavaScript glue code (WebGPU init, WASM loader, GLFW shim) |
| `galaxysim.wasm` | Compiled WebAssembly binary |

### Serving locally

Browsers require HTTP(S) to load WASM files -- you cannot open `galaxysim.html` directly from `file://`. Use any local HTTP server:

```bash
# Python (built-in)
python3 -m http.server -d build-web/src 8080

# Node.js
npx serve build-web/src -p 8080

# PHP
php -S localhost:8080 -t build-web/src
```

Then open **http://localhost:8080/galaxysim.html** in your browser.

### Browser compatibility

WebGPU is required. As of early 2026, support is as follows:

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| **Chrome** | 113+ | Supported | Enabled by default since Chrome 113 (May 2023) |
| **Edge** | 113+ | Supported | Chromium-based, same as Chrome |
| **Chrome Android** | 121+ | Supported | Android 12+ with compatible GPU |
| **Safari** | 18+ | Supported | macOS Sonoma / iOS 18+ with WebGPU feature flag |
| **Firefox** | Nightly | Experimental | Requires `dom.webgpu.enabled` in `about:config` |
| **Firefox (stable)** | -- | Not yet | WebGPU behind flag, not enabled by default |
| **Samsung Internet** | -- | Not yet | No WebGPU support |

**To check if your browser supports WebGPU**, open the browser console (F12) and run:

```javascript
console.log(!!navigator.gpu)  // true = supported
```

Or visit [webgpureport.org](https://webgpureport.org) for a detailed capability report.

#### Enabling WebGPU in Firefox

1. Open `about:config` in the address bar
2. Search for `dom.webgpu.enabled`
3. Set it to `true`
4. Restart Firefox

#### Enabling WebGPU in Safari

1. Open **Safari > Settings > Advanced**
2. Check **Show features for web developers**
3. Go to **Feature Flags** tab
4. Enable **WebGPU**
5. Restart Safari

### Build options for Emscripten

The CMake configuration automatically sets these Emscripten linker flags:

| Flag | Effect |
|------|--------|
| `-sASYNCIFY` | Enables async/await for WebGPU device initialization |
| `-sALLOW_MEMORY_GROWTH=1` | Lets WASM heap grow dynamically (needed for large N) |
| `-sFORCE_FILESYSTEM=1` | Enables Emscripten's virtual filesystem (for CSV export) |
| `-sUSE_GLFW=3` | Uses Emscripten's built-in GLFW 3 implementation |

The output is an `.html` file (set via `PROPERTIES SUFFIX ".html"`) that includes Emscripten's default shell with a canvas element and a text console.

### WebGPU backend variants

When building with Emscripten, the `WEBGPU_BACKEND` CMake variable supports two options:

```bash
# Default: uses Emscripten's built-in WebGPU bindings
emcmake cmake -B build-web -DWEBGPU_BACKEND=EMSCRIPTEN

# Alternative: Dawn's Emscripten port
emcmake cmake -B build-web -DWEBGPU_BACKEND=EMDAWNWEBGPU
```

The default (`EMSCRIPTEN`) uses the browser's native WebGPU implementation and is recommended. `EMDAWNWEBGPU` bundles Dawn compiled to WASM, which increases binary size but can be useful for testing Dawn-specific behavior.

### Differences from the native build

| Aspect | Native | Browser |
|--------|--------|---------|
| Main loop | `while (app.isRunning())` spin loop | `emscripten_set_main_loop_arg()` callback driven by `requestAnimationFrame` |
| GLFW | Fetched via FetchContent (native GLFW 3.4) | Emscripten's built-in GLFW 3 shim (`-sUSE_GLFW=3`) |
| Window size | 1280x720 fixed | Maps to HTML canvas size |
| `wgpuSurfacePresent` | Called explicitly each frame | Skipped (Emscripten presents automatically) |
| GPU backend | wgpu-native or Dawn | Browser's WebGPU implementation |
| `--headless` mode | Works (no window created) | Not applicable (always has canvas) |
| `--export` CSV | Writes to real filesystem | Writes to Emscripten virtual FS (can be downloaded via browser console) |
| Performance | Native GPU driver + CPU | WASM overhead + browser GPU layer |

### Downloading CSV exports in the browser

If you use `--export` in a browser build (via URL parameters or hardcoded config), the file is written to Emscripten's in-memory virtual filesystem. To retrieve it, open the browser console (F12) and run:

```javascript
// Read the file from Emscripten's virtual FS
var data = FS.readFile('/results.csv', { encoding: 'utf8' });

// Download it
var blob = new Blob([data], { type: 'text/csv' });
var a = document.createElement('a');
a.href = URL.createObjectURL(blob);
a.download = 'results.csv';
a.click();
```

### Performance tips for browser

- **Start with fewer particles**: 5,000--10,000 is a good starting range. WASM has overhead compared to native, and the octree build runs single-threaded in the browser.
- **Use a Chromium-based browser**: Chrome and Edge currently have the most mature WebGPU implementations.
- **Close other GPU-heavy tabs**: WebGPU shares GPU resources with the rest of the browser.
- **Increase theta**: Setting the Barnes-Hut opening angle to 1.0 or higher significantly reduces GPU work per frame at the cost of accuracy.
- **Check the Timing panel**: The ImGui timing breakdown works in the browser too -- use it to identify whether the bottleneck is tree build (CPU/WASM), force computation (GPU), or integration.

### Deploying to a web server

To host the simulation on a public server, copy the three build outputs and serve them with correct MIME types:

```bash
# Copy build artifacts
mkdir -p deploy
cp build-web/src/galaxysim.html deploy/index.html
cp build-web/src/galaxysim.js deploy/
cp build-web/src/galaxysim.wasm deploy/
```

Your web server must serve `.wasm` files with the correct MIME type. For **nginx**:

```nginx
types {
    application/wasm wasm;
}
```

For **Apache**, add to `.htaccess`:

```apache
AddType application/wasm .wasm
```

HTTPS is recommended (some browsers restrict WebGPU to secure contexts).

## Troubleshooting

**No GPU detected / WebGPU initialization fails**
- Ensure your GPU drivers support Vulkan (Linux/Windows) or Metal (macOS)
- On Linux, try running with `MESA_VK_DEVICE_SELECT=list` to check available devices

**Poor performance at high particle counts**
- Increase `--theta` to reduce octree traversal depth
- Increase `--softening` to prevent deep tree nodes from close encounters
- The octree build is single-threaded on CPU and becomes the bottleneck above ~50K particles

**Energy drift is large**
- Use `--integrator leapfrog` (default) instead of Euler
- Decrease `--dt` for tighter integration
- Decrease `--theta` for more accurate force computation

**Emscripten build fails**
- Ensure `emcmake` and `emcc` are on your PATH (`source emsdk/emsdk_env.sh`)
- Verify version: `emcc --version` (must be 3.1.50+)
- If you get linker errors about WebGPU symbols, your emsdk is too old -- run `./emsdk install latest && ./emsdk activate latest`

**Browser shows "WebGPU not supported" or blank page**
- Check WebGPU support: open the browser console and run `!!navigator.gpu`
- Chrome 113+ and Edge 113+ support WebGPU by default
- Firefox requires `dom.webgpu.enabled` in `about:config`
- Safari requires the WebGPU feature flag (see [Enabling WebGPU in Safari](#enabling-webgpu-in-safari))
- Some integrated GPUs (especially older Intel HD) may not be WebGPU-compatible even in supported browsers

**Browser console shows "Failed to create GPU adapter"**
- Your GPU or driver may not support the required WebGPU features
- On Linux in Chrome, try launching with `--enable-unsafe-webgpu --enable-features=Vulkan`
- On macOS, WebGPU uses Metal -- ensure macOS 13+ (Ventura) or later

**WASM file fails to load (MIME type error)**
- Your HTTP server must serve `.wasm` files as `application/wasm`
- `file://` URLs do not work -- you must use an HTTP server
- See the [Deploying to a web server](#deploying-to-a-web-server) section for server configuration

**Window opens but stays black**
- Click **Play** in the GUI panel -- the simulation starts paused by default
