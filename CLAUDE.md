# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Barnes-Hut N-body galaxy simulator (up to 100K particles) built with C++20 and WebGPU.

## Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run interactive
./build/src/galaxysim

# Run headless batch mode
./build/src/galaxysim --headless --scenario plummer --N 1000 --steps 5000 --export results.csv

# Emscripten build
emcmake cmake -B build-web
cmake --build build-web
```

All dependencies are fetched automatically via `FetchContent` (WebGPU-distribution, glfw, glfw3webgpu, spdlog, imgui, glm). No manual dependency installation needed.

Backend selection: `-DWEBGPU_BACKEND=WGPU` (default) or `DAWN` or `EMSCRIPTEN`.

## Architecture

### GPU-Primary Architecture

All physics runs on the GPU. The CPU tree path reads back positions via staging buffers when it needs to build an octree. Diagnostics always use GPU readback. Bind groups are cached and only recreated when particle count changes.

### Simulation Step (KDK Leapfrog)

1. **Half-kick** (CPU loop + GPU kick shader): `v += a * dt/2`
2. **Drift** (CPU loop + GPU drift shader): `x += v * dt`
3. **Octree build** (CPU only): `Octree::build()` from CPU positions → flatten to `OctreeNode[]` → upload to GPU
4. **Force computation** (CPU Barnes-Hut traversal + GPU force shader): both traverse the same octree
5. **Second half-kick** (CPU loop + GPU kick shader): `v += a * dt/2`

The Euler integrator is preserved as a `--integrator euler` fallback with the original force→integrate sequence.

### Shader Embedding

All WGSL shaders are **embedded as C string literals** in `simulation.cpp` (4 compute shaders) and `particle_renderer.cpp` (1 render shader). There are no separate `.wgsl` files.

### Rendering Pipeline

Particles are rendered as instanced billboard quads (6 vertices each) with additive blending. Positions and colors are read directly from GPU storage buffers via `@builtin(instance_index)` — no vertex buffer indirection.

### CLI Configuration

```
--scenario    twobody | plummer | disk     (default: disk)
--integrator  euler | leapfrog             (default: leapfrog)
--N           particle count               (default: 10000)
--dt / --softening / --theta / --seed      physics parameters
--steps       step limit (0=interactive)
--export      CSV output path
--headless    batch mode (no window)
```

### Key Files

| File | Role |
|------|------|
| `simulation.cpp` | Physics engine: shaders, scenarios, integrators, GPU step logic |
| `octree.cpp` | CPU octree: recursive insert, COM propagation, flatten to GPU layout |
| `particle_renderer.cpp` | WebGPU render pipeline with embedded WGSL vertex/fragment shader |
| `main.cpp` | Application class (interactive) + `runHeadless()` (batch) |
| `diagnostics.cpp` | Conservation metrics: kinetic/potential energy, momentum, drift |
| `config.cpp` | CLI argument parser |
| `exporter.cpp` | Per-step CSV export |

### OctreeNode GPU Layout

```cpp
struct OctreeNode {
    glm::vec4 centerOfMass;  // xyz = COM, w = total mass
    glm::vec4 bounds;        // xyz = box center, w = half-width
    int32_t children[8];     // -1 = empty
};
```

The GPU force shader traverses this with an explicit stack (depth 64) using the Barnes-Hut opening criterion: `halfWidth² / distSq < θ²`.

### BVHNodeGPU Layout (64 bytes, float32 bounds)

```cpp
struct BVHNodeGPU {
    glm::vec4 centerOfMass;  // xyz = COM, w = mass
    glm::vec4 boundsMin;     // xyz = AABB min
    glm::vec4 boundsMax;     // xyz = AABB max
    int32_t left, right, parent, particleIdx;
};
```

Float32 bounds avoid quantization artifacts that caused force accuracy degradation in dense particle clusters. The BVH force shader uses a mass-adaptive opening criterion: `halfExtent = comToCornerRadius * (1 + 0.6 * log2(mass))`, which compensates for the BVH's deeper binary tree (log2 N vs octree's log8 N levels).

### Params Uniform Buffer Layout (32 bytes)

```
offset 0:  f32 dt
offset 4:  f32 softening
offset 8:  f32 theta
offset 12: u32 numParticles
offset 16: u32 nodeCount
offset 20: f32[3] padding
```

This struct must match between C++ (`alignas(16)`) and WGSL exactly.

## Conventions

- WebGPU C API throughout (raw `WGPUDevice`, `WGPUBuffer`, etc.) — no C++ wrapper libraries
- Supports native (wgpu/Dawn) and WASM (Emscripten) from the same CMakeLists.txt
- Each source file is built as its own static library in CMake, then linked into the final executable
- Bind groups are cached and reused; only recreated when particle count changes
- `Buffer::upload()` auto-resizes if capacity is exceeded
