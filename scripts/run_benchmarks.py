#!/usr/bin/env python3
"""Run benchmark experiments for the galaxy simulator.

Usage:
    python scripts/run_benchmarks.py                       # run all experiments
    python scripts/run_benchmarks.py --experiment P1       # run only Priority 1
    python scripts/run_benchmarks.py --binary build-dawn/src/galaxysim  # use Dawn backend
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DEFAULT_BINARY = os.path.join(PROJECT_ROOT, "build", "src", "galaxysim")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "benchmarks")

EXPERIMENTS = {
    "P1_plummer": {
        "description": "Priority 1: Plummer sphere N-scaling (50 warmup + 100 measured)",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "scenario": "plummer",
        "extra_args": ["--sync-timing"],
    },
    "P2_lbvh": {
        "description": "Priority 2: Per-pass LBVH breakdown",
        "Ns": [1000, 5000, 10000, 50000, 100000],
        "steps": 150,
        "scenario": "plummer",
        "extra_args": ["--benchmark-passes"],
    },
    "P5_direct": {
        "description": "Priority 5: Direct O(N^2) for crossover comparison",
        "Ns": [100, 500, 1000, 2000, 5000],
        "steps": 150,
        "scenario": "plummer",
        "force_method": "direct",
        "extra_args": ["--sync-timing"],
    },
    "P5_tree": {
        "description": "Priority 5: Tree for crossover comparison",
        "Ns": [100, 500, 1000, 2000, 5000],
        "steps": 150,
        "scenario": "plummer",
        "force_method": "tree",
        "extra_args": ["--sync-timing"],
    },
}


def run_single(binary, experiment_name, config, n, results_dir, backend_tag=""):
    """Run a single benchmark configuration."""
    tag = f"_{backend_tag}" if backend_tag else ""
    force = config.get("force_method", "tree")
    csv_name = f"{experiment_name}{tag}_N{n}.csv"
    csv_path = os.path.join(results_dir, csv_name)

    cmd = [
        binary,
        "--headless",
        "--scenario", config["scenario"],
        "--N", str(n),
        "--steps", str(config["steps"]),
        "--force-method", force,
        "--export", csv_path,
        *config.get("extra_args", []),
    ]

    print(f"  N={n:>6} -> {csv_name} ... ", end="", flush=True)
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr.strip()[:200]}")
        return False

    print(f"OK ({elapsed:.1f}s)")
    return True


def run_experiment(binary, name, config, results_dir, backend_tag=""):
    """Run all N values for one experiment."""
    print(f"\n{'='*60}")
    print(f"  {name}: {config['description']}")
    print(f"{'='*60}")

    successes = 0
    for n in config["Ns"]:
        if run_single(binary, name, config, n, results_dir, backend_tag):
            successes += 1

    total = len(config["Ns"])
    print(f"  Completed: {successes}/{total}")
    return successes == total


def main():
    parser = argparse.ArgumentParser(description="Run galaxy simulator benchmarks")
    parser.add_argument("--binary", default=DEFAULT_BINARY,
                        help="Path to galaxysim binary")
    parser.add_argument("--experiment", "-e", nargs="*",
                        help="Experiment(s) to run (e.g. P1 P2). Default: all")
    parser.add_argument("--backend-tag", default="",
                        help="Tag for backend (e.g. 'dawn', 'wgpu')")
    parser.add_argument("--output-dir", default=RESULTS_DIR,
                        help="Output directory for CSV files")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Error: binary not found at {args.binary}")
        print("Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
        sys.exit(1)

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

    print(f"Binary: {args.binary}")
    print(f"Output: {args.output_dir}")
    print(f"Experiments: {', '.join(selected.keys())}")

    all_ok = True
    t_total = time.time()
    for name, config in selected.items():
        if not run_experiment(args.binary, name, config, args.output_dir,
                              args.backend_tag):
            all_ok = False

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
