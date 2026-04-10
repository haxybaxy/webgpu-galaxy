#!/usr/bin/env python3
"""Analyze benchmark results and generate figures for the capstone paper.

Usage:
    python scripts/analyze_benchmarks.py                    # generate all figures
    python scripts/analyze_benchmarks.py --fig 1 2          # specific figures only
    python scripts/analyze_benchmarks.py --results-dir path # custom results dir

Requires: matplotlib, numpy, scipy
    pip install matplotlib numpy scipy
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
BENCH_DIR = os.path.join(PROJECT_ROOT, "results", "benchmarks")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
WEB_LOG = os.path.join(PROJECT_ROOT, "web_results.txt")

WARMUP = 50  # steps to discard

# Academic style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Helpers ──────────────────────────────────────────────────────

def load_csv(path, warmup=WARMUP):
    """Load a benchmark CSV, skip warmup rows, return list of dicts."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[warmup:]


def compute_stats(values):
    """Return dict with mean, std, ci95, cv for an array of floats."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci95 = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "cv": cv, "n": n}


def load_experiment(prefix, ns, warmup=WARMUP, results_dir=BENCH_DIR):
    """Load CSVs for an experiment, return dict of N -> rows."""
    data = {}
    for n in ns:
        path = os.path.join(results_dir, f"{prefix}_N{n}.csv")
        if os.path.exists(path):
            data[n] = load_csv(path, warmup)
        else:
            print(f"  Warning: missing {path}")
    return data


def timing_stats(rows, column):
    """Compute stats for a timing column across rows."""
    values = [float(r[column]) for r in rows]
    return compute_stats(values)


def total_ms(row):
    """Total step time from a row."""
    return float(row["tree_build_ms"]) + float(row["force_ms"]) + float(row["integrate_ms"])


# ── Figure 1: N-scaling log-log (Priority 1) ────────────────────

def fig1_plummer_scaling(results_dir, fig_dir):
    """Log-log plot of ms/step vs N for Plummer sphere."""
    ns = [1000, 5000, 10000, 50000, 100000]
    data = load_experiment("P1_plummer", ns, results_dir=results_dir)
    if not data:
        print("  Skipping fig1: no P1 data found")
        return

    ns_found = sorted(data.keys())
    total_stats = [compute_stats([total_ms(r) for r in data[n]]) for n in ns_found]
    tree_stats = [timing_stats(data[n], "tree_build_ms") for n in ns_found]
    force_stats = [timing_stats(data[n], "force_ms") for n in ns_found]

    fig, ax = plt.subplots()

    # Data lines with error bars
    ax.errorbar(ns_found, [s["mean"] for s in total_stats],
                yerr=[s["ci95"] for s in total_stats],
                fmt="o-", capsize=3, label="Total", color="black", markersize=4)
    ax.errorbar(ns_found, [s["mean"] for s in tree_stats],
                yerr=[s["ci95"] for s in tree_stats],
                fmt="s--", capsize=3, label="Tree build", color="tab:blue", markersize=4)
    ax.errorbar(ns_found, [s["mean"] for s in force_stats],
                yerr=[s["ci95"] for s in force_stats],
                fmt="^--", capsize=3, label="Force", color="tab:red", markersize=4)

    # O(N log N) reference line anchored at the smallest N
    n0 = ns_found[0]
    t0 = total_stats[0]["mean"]
    ref_ns = np.array(ns_found, dtype=float)
    ref_line = t0 * (ref_ns * np.log2(ref_ns)) / (n0 * np.log2(n0))
    ax.plot(ref_ns, ref_line, ":", color="gray", alpha=0.7, label=r"$O(N \log N)$ ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Plummer Sphere: Step Time vs Particle Count")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_plummer_scaling")
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")

    # Print summary table
    print("\n  N-Scaling Summary (ms/step):")
    print(f"  {'N':>7}  {'mean':>8}  {'std':>8}  {'95% CI':>8}  {'CV':>6}")
    for n, s in zip(ns_found, total_stats):
        print(f"  {n:>7}  {s['mean']:>8.3f}  {s['std']:>8.3f}  {s['ci95']:>8.3f}  {s['cv']:>6.3f}")


# ── Figure 2: LBVH stacked bar chart (Priority 2) ───────────────

def fig2_lbvh_breakdown(results_dir, fig_dir):
    """Stacked bar chart of LBVH pass timing breakdown."""
    ns = [1000, 5000, 10000, 50000, 100000]
    data = load_experiment("P2_lbvh", ns, results_dir=results_dir)
    if not data:
        print("  Skipping fig2: no P2 data found")
        return

    ns_found = sorted(data.keys())
    pass_names = ["bbox_ms", "morton_ms", "radix_sort_ms",
                  "karras_ms", "leaf_init_ms", "aggregate_ms"]
    labels = ["Bbox reduce", "Morton codes", "Radix sort",
              "Karras build", "Leaf init", "Aggregation"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    # Compute mean per pass per N
    means = {p: [] for p in pass_names}
    for n in ns_found:
        for p in pass_names:
            s = timing_stats(data[n], p)
            means[p].append(s["mean"])

    fig, ax = plt.subplots()
    x = np.arange(len(ns_found))
    width = 0.6
    bottom = np.zeros(len(ns_found))

    for p, label, color in zip(pass_names, labels, colors):
        vals = np.array(means[p])
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        bottom += vals

    # Annotate dominant pass at each N
    for i, n in enumerate(ns_found):
        dominant_idx = max(range(len(pass_names)),
                          key=lambda j: means[pass_names[j]][i])
        dominant_val = means[pass_names[dominant_idx]][i]
        total_val = bottom[i]
        pct = dominant_val / total_val * 100 if total_val > 0 else 0
        ax.text(i, total_val + 0.1, f"{labels[dominant_idx].split()[0]}\n{pct:.0f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ns_found], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms")
    ax.set_title("LBVH Construction: Per-Pass Breakdown")
    ax.legend(fontsize=7, loc="upper left")

    path = os.path.join(fig_dir, "fig_lbvh_breakdown")
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")

    # Print breakdown table
    print("\n  LBVH Pass Breakdown (mean ms):")
    header = f"  {'N':>7}" + "".join(f"  {l[:8]:>8}" for l in labels) + f"  {'Total':>8}"
    print(header)
    for i, n in enumerate(ns_found):
        row = f"  {n:>7}"
        t = 0
        for p in pass_names:
            v = means[p][i]
            row += f"  {v:>8.3f}"
            t += v
        row += f"  {t:>8.3f}"
        print(row)


# ── Figure 3: Native vs browser (Priority 4) ────────────────────

def parse_web_logs(web_log_path):
    """Parse browser console logs, return dict of N -> {wallclock_ms, gpu_ms}."""
    import re
    from datetime import datetime

    if not os.path.exists(web_log_path):
        return {}

    with open(web_log_path) as f:
        text = f.read()

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
        m = re.search(r"(\d+) particles", run[0])
        if not m:
            continue
        n = int(m.group(1))

        step_lines = [l for l in run if re.search(r"Step \d+/\d+", l)]
        if len(step_lines) < 2:
            continue

        gpu_totals = []
        for sl in step_lines:
            m = re.search(r"total=(\d+\.\d+)ms", sl)
            if m:
                gpu_totals.append(float(m.group(1)))

        def parse_ts(line):
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", line)
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None

        first_ts = parse_ts(step_lines[0])
        last_ts = parse_ts(step_lines[-1])
        first_step = int(re.search(r"Step (\d+)/", step_lines[0]).group(1))
        last_step = int(re.search(r"Step (\d+)/", step_lines[-1]).group(1))
        delta_steps = last_step - first_step

        if delta_steps > 0 and first_ts and last_ts:
            wallclock_ms = (last_ts - first_ts).total_seconds() * 1000 / delta_steps
        else:
            wallclock_ms = 0

        results[n] = {
            "wallclock_ms": wallclock_ms,
            "gpu_ms": sum(gpu_totals) / len(gpu_totals) if gpu_totals else 0,
        }
    return results


def fig3_native_vs_browser(results_dir, fig_dir, web_log_path):
    """Grouped bar chart comparing native vs browser performance."""
    ns = [1000, 5000, 10000, 50000, 100000]
    native_data = load_experiment("P1_plummer", ns, results_dir=results_dir)
    web_data = parse_web_logs(web_log_path)

    if not native_data:
        print("  Skipping fig3: no P1 native data found")
        return
    if not web_data:
        print("  Skipping fig3: no web_results.txt found")
        return

    # Find Ns present in both
    common_ns = sorted(set(native_data.keys()) & set(web_data.keys()))
    if not common_ns:
        print("  Skipping fig3: no overlapping N values between native and web")
        return

    native_means = [compute_stats([total_ms(r) for r in native_data[n]])["mean"]
                    for n in common_ns]
    web_means = [web_data[n]["wallclock_ms"] for n in common_ns]

    fig, ax = plt.subplots()
    x = np.arange(len(common_ns))
    width = 0.35

    bars_native = ax.bar(x - width / 2, native_means, width, label="Native (wgpu)",
                         color="tab:blue")
    bars_web = ax.bar(x + width / 2, web_means, width, label="Browser (Chrome)",
                      color="tab:orange", hatch="//")

    # Shade the overhead
    for i in range(len(common_ns)):
        if web_means[i] > native_means[i]:
            ax.bar(x[i] + width / 2, web_means[i] - native_means[i], width,
                   bottom=native_means[i], color="tab:red", alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in common_ns], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Native vs Browser Performance")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_native_vs_browser")
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


# ── Figure 4: Cross-backend (Priority 3, nice-to-have) ──────────

def fig4_cross_backend(results_dir, fig_dir):
    """Grouped bar chart comparing backends."""
    ns = [1000, 10000, 100000]
    backends = [
        ("wgpu", "P1_plummer_wgpu"),
        ("Dawn", "P1_plummer_dawn"),
        ("Chrome", "P1_plummer_chrome"),
        ("Safari", "P1_plummer_safari"),
    ]

    backend_data = {}
    for label, prefix in backends:
        data = load_experiment(prefix, ns, results_dir=results_dir)
        if data:
            backend_data[label] = data

    if len(backend_data) < 2:
        # Try without suffix (default wgpu)
        data = load_experiment("P1_plummer", ns, results_dir=results_dir)
        if data and "wgpu" not in backend_data:
            backend_data["wgpu"] = data

    if len(backend_data) < 2:
        print("  Skipping fig4: need at least 2 backends")
        return

    fig, ax = plt.subplots()
    n_backends = len(backend_data)
    width = 0.8 / n_backends
    x = np.arange(len(ns))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for i, (label, data) in enumerate(backend_data.items()):
        means = []
        for n in ns:
            if n in data:
                means.append(compute_stats([total_ms(r) for r in data[n]])["mean"])
            else:
                means.append(0)
        offset = (i - n_backends / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Cross-Backend Comparison")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_cross_backend")
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


# ── Figure 5: Direct vs tree crossover (Priority 5) ─────────────

def fig5_crossover(results_dir, fig_dir):
    """Direct vs tree crossover plot."""
    ns = [100, 500, 1000, 2000, 5000, 10000, 20000]

    direct_data = load_experiment("P5_direct", ns, results_dir=results_dir)
    tree_data = load_experiment("P5_tree", ns, results_dir=results_dir)

    if not direct_data or not tree_data:
        print("  Skipping fig5: need both P5_direct and P5_tree data")
        return

    common_ns = sorted(set(direct_data.keys()) & set(tree_data.keys()))
    direct_means = [compute_stats([total_ms(r) for r in direct_data[n]])["mean"]
                    for n in common_ns]
    tree_means = [compute_stats([total_ms(r) for r in tree_data[n]])["mean"]
                  for n in common_ns]

    fig, ax = plt.subplots()

    ax.plot(common_ns, direct_means, "o-", label=r"Direct $O(N^2)$", color="tab:red")
    ax.plot(common_ns, tree_means, "s-", label=r"Tree $O(N \log N)$", color="tab:blue")

    # O(N log N) reference
    n0 = common_ns[0]
    t0 = tree_means[0]
    ref_ns = np.array(common_ns, dtype=float)
    ref_nlogn = t0 * (ref_ns * np.log2(ref_ns)) / (n0 * np.log2(n0))
    ax.plot(ref_ns, ref_nlogn, ":", color="tab:blue", alpha=0.4,
            label=r"$O(N \log N)$ ref")

    # O(N^2) reference
    d0 = direct_means[0]
    ref_n2 = d0 * (ref_ns / n0) ** 2
    ax.plot(ref_ns, ref_n2, ":", color="tab:red", alpha=0.4, label=r"$O(N^2)$ ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Direct vs Tree: Crossover Point")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_crossover")
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


# ── Summary stats CSV ────────────────────────────────────────────

def write_summary_csv(results_dir, fig_dir):
    """Write summary statistics CSV for LaTeX tables."""
    ns = [1000, 5000, 10000, 50000, 100000]
    data = load_experiment("P1_plummer", ns, results_dir=results_dir)
    if not data:
        return

    path = os.path.join(fig_dir, "summary_stats.csv")
    with open(path, "w") as f:
        f.write("N,component,mean_ms,std_ms,ci95_ms,cv\n")
        for n in sorted(data.keys()):
            rows = data[n]
            for col, label in [("tree_build_ms", "tree_build"),
                               ("force_ms", "force"),
                               ("integrate_ms", "integrate")]:
                s = timing_stats(rows, col)
                f.write(f"{n},{label},{s['mean']:.4f},{s['std']:.4f},"
                        f"{s['ci95']:.4f},{s['cv']:.4f}\n")
            # Total
            s = compute_stats([total_ms(r) for r in rows])
            f.write(f"{n},total,{s['mean']:.4f},{s['std']:.4f},"
                    f"{s['ci95']:.4f},{s['cv']:.4f}\n")

    # Also write LBVH breakdown if available
    lbvh_data = load_experiment("P2_lbvh", ns, results_dir=results_dir)
    if lbvh_data:
        lbvh_path = os.path.join(fig_dir, "lbvh_breakdown_stats.csv")
        pass_cols = ["bbox_ms", "morton_ms", "radix_sort_ms",
                     "karras_ms", "leaf_init_ms", "aggregate_ms"]
        with open(lbvh_path, "w") as f:
            f.write("N,pass,mean_ms,std_ms,ci95_ms,cv\n")
            for n in sorted(lbvh_data.keys()):
                for col in pass_cols:
                    s = timing_stats(lbvh_data[n], col)
                    label = col.replace("_ms", "")
                    f.write(f"{n},{label},{s['mean']:.4f},{s['std']:.4f},"
                            f"{s['ci95']:.4f},{s['cv']:.4f}\n")
        print(f"  Saved: {lbvh_path}")

    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmarks and generate figures")
    parser.add_argument("--results-dir", default=BENCH_DIR,
                        help="Directory with benchmark CSVs")
    parser.add_argument("--fig-dir", default=FIG_DIR,
                        help="Output directory for figures")
    parser.add_argument("--web-log", default=WEB_LOG,
                        help="Path to web_results.txt for browser comparison")
    parser.add_argument("--fig", nargs="*", type=int,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    figs = args.fig or [1, 2, 3, 4, 5]

    if 1 in figs:
        print("\nFigure 1: Plummer N-scaling (log-log)")
        fig1_plummer_scaling(args.results_dir, args.fig_dir)

    if 2 in figs:
        print("\nFigure 2: LBVH pass breakdown (stacked bar)")
        fig2_lbvh_breakdown(args.results_dir, args.fig_dir)

    if 3 in figs:
        print("\nFigure 3: Native vs browser")
        fig3_native_vs_browser(args.results_dir, args.fig_dir, args.web_log)

    if 4 in figs:
        print("\nFigure 4: Cross-backend comparison")
        fig4_cross_backend(args.results_dir, args.fig_dir)

    if 5 in figs:
        print("\nFigure 5: Direct vs tree crossover")
        fig5_crossover(args.results_dir, args.fig_dir)

    print("\nSummary statistics:")
    write_summary_csv(args.results_dir, args.fig_dir)


if __name__ == "__main__":
    main()
