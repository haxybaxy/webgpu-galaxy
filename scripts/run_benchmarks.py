#!/usr/bin/env python3
"""Run benchmark experiments for the galaxy simulator.

Usage:
    python scripts/run_benchmarks.py                          # run all experiments
    python scripts/run_benchmarks.py -e wgpu_sync wgpu_direct # run specific experiments
    python scripts/run_benchmarks.py -e dawn_sync             # Dawn backend only
    python scripts/run_benchmarks.py -e theta_sweep           # theta parameter sweep

Experiments and which paper tables they feed:
    wgpu_sync          — wgpu-native tree, --sync-timing    → Tab1, Tab3(tree), Tab4(WebGPU), Tab5(wgpu), Tab6(Native)
    wgpu_direct        — wgpu-native direct, --sync-timing   → Tab3(direct)
    wgpu_lbvh          — wgpu-native tree, --benchmark-passes → Tab2 (LBVH breakdown only)
    dawn_sync          — Dawn tree, --sync-timing             → Tab5(Dawn)
    theta_sweep        — wgpu-native tree, varying theta      → Tab7
    twobody_sweep      — two-body dt sweep, tree, --sync-timing → Group 1 validation
    dt_sweep           — plummer, varying dt                   → Group 2c
    softening_sweep    — plummer, varying softening            → Group 2e
    disk_sync          — disk scenario, --sync-timing          → Group 3a

Browser experiments (Chrome, Safari) must be run manually and placed in results/benchmarks/:
    chrome_sync_N{N}.csv  — N={1000,5000,10000,50000,100000} → Tab5(Chrome), Tab6(Chrome)
    safari_sync_N{N}.csv  — N={1000,10000,100000}             → Tab5(Safari)
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DEFAULT_WGPU_BINARY = os.path.join(PROJECT_ROOT, "build", "src", "galaxysim")
DEFAULT_DAWN_BINARY = os.path.join(PROJECT_ROOT, "build-dawn", "src", "galaxysim")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "benchmarks")

# Common physics parameters (METHODOLOGY.md protocol)
COMMON_PARAMS = {
    "scenario": "plummer",
    "dt": "0.001",
    "softening": "0.5",
    "theta": "0.75",
    "seed": "42",
}

EXPERIMENTS = {
    "wgpu_sync": {
        "description": "wgpu-native tree (--sync-timing) → Tab1,3,4,5,6",
        "binary": "wgpu",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
    },
    "wgpu_direct": {
        "description": "wgpu-native direct (--sync-timing) → Tab3(direct)",
        "binary": "wgpu",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "force_method": "direct",
        "extra_args": ["--sync-timing"],
    },
    "wgpu_lbvh": {
        "description": "wgpu-native tree (--benchmark-passes) → Tab2 LBVH breakdown",
        "binary": "wgpu",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "force_method": "tree",
        "extra_args": ["--benchmark-passes"],
    },
    "dawn_sync": {
        "description": "Dawn tree (--sync-timing) → Tab5(Dawn)",
        "binary": "dawn",
        "Ns": [1000, 10000, 100000],
        "steps": 150,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
    },
    "theta_sweep": {
        "description": "Theta parameter sweep (--sync-timing) → Tab7",
        "binary": "wgpu",
        "Ns": [5000],
        "steps": 5000,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
        "theta_values": [0.3, 0.5, 0.7, 1.0],
    },
    "twobody_sweep": {
        "description": "Two-body dt sweep, tree (--sync-timing) → Group 1",
        "binary": "wgpu",
        "Ns": [2],
        "steps": 5000,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
        "scenario": "twobody",
        "dt_values": [0.0001, 0.0005, 0.001, 0.005],
    },
    "dt_sweep": {
        "description": "Timestep sweep (--sync-timing) → Group 2c",
        "binary": "wgpu",
        "Ns": [5000],
        "steps": 5000,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
        "dt_values": [0.0001, 0.0005, 0.001, 0.005],
    },
    "softening_sweep": {
        "description": "Softening sweep (--sync-timing) → Group 2e",
        "binary": "wgpu",
        "Ns": [5000],
        "steps": 5000,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
        "softening_values": [0.1, 0.25, 0.5, 1.0, 2.0],
    },
    "disk_sync": {
        "description": "Disk N-scaling (--sync-timing) → Group 3a",
        "binary": "wgpu",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
        "scenario": "disk",
    },
}


def run_single(binary, csv_path, n, steps, force_method, extra_args,
               theta=None, scenario=None, dt=None, softening=None):
    """Run a single benchmark configuration."""
    params = dict(COMMON_PARAMS)
    if theta is not None:
        params["theta"] = str(theta)
    if scenario is not None:
        params["scenario"] = scenario
    if dt is not None:
        params["dt"] = str(dt)
    if softening is not None:
        params["softening"] = str(softening)

    cmd = [
        binary,
        "--headless",
        "--scenario", params["scenario"],
        "--N", str(n),
        "--steps", str(steps),
        "--dt", params["dt"],
        "--softening", params["softening"],
        "--theta", params["theta"],
        "--seed", params["seed"],
        "--force-method", force_method,
        "--export", csv_path,
        *extra_args,
    ]

    csv_name = os.path.basename(csv_path)
    print(f"  {csv_name} ... ", end="", flush=True)
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr.strip()[:300]}")
        return False

    size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
    if size < 500:
        print(f"WARNING: output only {size}B ({elapsed:.1f}s)")
        return False

    print(f"OK ({elapsed:.1f}s)")
    return True


def run_experiment(binaries, name, config, results_dir):
    """Run all configurations for one experiment."""
    print(f"\n{'='*60}")
    print(f"  {name}: {config['description']}")
    print(f"{'='*60}")

    binary = binaries[config["binary"]]
    if not os.path.isfile(binary):
        print(f"  SKIP: binary not found at {binary}")
        return False

    successes = 0
    total = 0
    scenario = config.get("scenario")

    theta_values = config.get("theta_values")
    dt_values = config.get("dt_values")
    softening_values = config.get("softening_values")

    if theta_values:
        for theta in theta_values:
            for n in config["Ns"]:
                csv_name = f"theta_{theta}_N{n}.csv"
                csv_path = os.path.join(results_dir, csv_name)
                total += 1
                if run_single(binary, csv_path, n, config["steps"],
                              config["force_method"], config["extra_args"],
                              theta=theta, scenario=scenario):
                    successes += 1
    elif dt_values:
        for dt in dt_values:
            for n in config["Ns"]:
                csv_name = f"dt_{dt}_N{n}.csv"
                csv_path = os.path.join(results_dir, csv_name)
                total += 1
                if run_single(binary, csv_path, n, config["steps"],
                              config["force_method"], config["extra_args"],
                              dt=dt, scenario=scenario):
                    successes += 1
    elif softening_values:
        for soft in softening_values:
            for n in config["Ns"]:
                csv_name = f"softening_{soft}_N{n}.csv"
                csv_path = os.path.join(results_dir, csv_name)
                total += 1
                if run_single(binary, csv_path, n, config["steps"],
                              config["force_method"], config["extra_args"],
                              softening=soft, scenario=scenario):
                    successes += 1
    else:
        for n in config["Ns"]:
            csv_name = f"{name}_N{n}.csv"
            csv_path = os.path.join(results_dir, csv_name)
            total += 1
            if run_single(binary, csv_path, n, config["steps"],
                          config["force_method"], config["extra_args"],
                          scenario=scenario):
                successes += 1

    print(f"  Completed: {successes}/{total}")
    return successes == total


def main():
    parser = argparse.ArgumentParser(description="Run galaxy simulator benchmarks")
    parser.add_argument("--wgpu-binary", default=DEFAULT_WGPU_BINARY,
                        help="Path to wgpu galaxysim binary")
    parser.add_argument("--dawn-binary", default=DEFAULT_DAWN_BINARY,
                        help="Path to Dawn galaxysim binary")
    parser.add_argument("--experiment", "-e", nargs="*",
                        help="Experiment(s) to run (default: all)")
    parser.add_argument("--output-dir", default=RESULTS_DIR,
                        help="Output directory for CSV files")
    args = parser.parse_args()

    binaries = {
        "wgpu": args.wgpu_binary,
        "dawn": args.dawn_binary,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter experiments
    if args.experiment:
        selected = {}
        for prefix in args.experiment:
            for name, config in EXPERIMENTS.items():
                if name.startswith(prefix):
                    selected[name] = config
        if not selected:
            print(f"No experiments match: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
    else:
        selected = EXPERIMENTS

    print(f"wgpu binary: {binaries['wgpu']}")
    print(f"Dawn binary: {binaries['dawn']}")
    print(f"Output:      {args.output_dir}")
    print(f"Experiments: {', '.join(selected.keys())}")

    all_ok = True
    t_total = time.time()
    for name, config in selected.items():
        if not run_experiment(binaries, name, config, args.output_dir):
            all_ok = False

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")

    if not all_ok:
        print("\nSome experiments failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
