#!/usr/bin/env python3
"""Generate figures for the capstone paper from benchmark CSV data.

Usage:
    python3 scripts/analyze_benchmarks.py              # generate all figures
    python3 scripts/analyze_benchmarks.py --fig 1 2    # specific figures only
    python3 scripts/analyze_benchmarks.py --print-tables  # print table data to stdout

Requires: matplotlib, numpy, pandas
    pip install matplotlib numpy pandas

Figures:
    1: fig_n_scaling_plummer  — Log-log ms/step vs N (Plummer sphere)
    2: fig_crossover          — Dual-axis: runtime + energy drift, direct vs tree
    3: fig_web_native         — Native (wgpu) vs Browser (Chrome) ms/step
    4: fig_lbvh_breakdown     — Per-pass stacked bar chart (LBVH at each N)
    5: fig_cross_backend      — Grouped bar: wgpu vs Dawn vs Chrome vs Safari

Data is read from CSV files in results/benchmarks/ (not hardcoded).
Protocol: first 50 steps are warmup (discarded), steps 51-150 are measured.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
BENCH_DIR = os.path.join(PROJECT_ROOT, "results", "benchmarks")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

WARMUP_STEPS = 50

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


# ═══════════════════════════════════════════════════════════════════
# CSV loading and statistics
# ═══════════════════════════════════════════════════════════════════

def load_csv(path):
    """Load a benchmark CSV, discarding warmup steps."""
    df = pd.read_csv(path)
    return df[df["step"] > WARMUP_STEPS].copy()


def compute_stats(series):
    """Compute mean, std, CV, CI95 for a pandas Series."""
    mean = series.mean()
    std = series.std()
    n = len(series)
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else float("inf"),
        "ci95": 1.96 * std / np.sqrt(n) if n > 0 else 0,
    }


def total_ms(df):
    """Compute total ms/step from component columns."""
    cols = ["tree_build_ms", "force_ms", "integrate_ms"]
    available = [c for c in cols if c in df.columns]
    return df[available].sum(axis=1)


def load_experiment(prefix, ns, bench_dir=BENCH_DIR):
    """Load all N values for an experiment prefix.

    Returns dict: {N: DataFrame} for CSVs that exist.
    Missing files are reported and skipped.
    """
    data = {}
    missing = []
    for n in ns:
        path = os.path.join(bench_dir, f"{prefix}_N{n}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 200:
            data[n] = load_csv(path)
        else:
            missing.append(n)
    if missing:
        print(f"  Warning: missing {prefix} for N={missing}")
    return data


# ═══════════════════════════════════════════════════════════════════
# Figure generators
# ═══════════════════════════════════════════════════════════════════

def fig1_n_scaling_plummer(fig_dir, bench_dir):
    """Log-log plot of ms/step vs N with component breakdown."""
    ns = [1000, 5000, 10000, 50000, 100000]
    data = load_experiment("wgpu_sync", ns, bench_dir)
    if not data:
        print("  SKIP: no wgpu_sync data")
        return

    plot_ns, totals, stds, trees, forces = [], [], [], [], []
    for n in ns:
        if n not in data:
            continue
        df = data[n]
        ms = total_ms(df)
        plot_ns.append(n)
        totals.append(ms.mean())
        stds.append(ms.std())
        trees.append(df["tree_build_ms"].mean())
        forces.append(df["force_ms"].mean())

    plot_ns = np.array(plot_ns)
    fig, ax = plt.subplots()

    ax.errorbar(plot_ns, totals, yerr=stds, fmt="o-", capsize=3,
                label="Total", color="black", markersize=4)
    ax.plot(plot_ns, trees, "s--", label="Tree build", color="tab:blue", markersize=4)
    ax.plot(plot_ns, forces, "^--", label="Force", color="tab:red", markersize=4)

    # O(N log N) reference
    n0, t0 = plot_ns[0], totals[0]
    ref = t0 * (plot_ns * np.log2(plot_ns)) / (n0 * np.log2(n0))
    ax.plot(plot_ns, ref, ":", color="gray", alpha=0.7, label=r"$O(N \log N)$ ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Plummer Sphere: Step Time vs Particle Count")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_n_scaling_plummer")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.{{pdf,png}}")


def fig2_crossover(fig_dir, bench_dir):
    """Dual-axis crossover: runtime (left y) + energy drift (right y)."""
    ns = [1000, 5000, 10000, 50000, 100000]
    tree_data = load_experiment("wgpu_sync", ns, bench_dir)
    direct_data = load_experiment("wgpu_direct", ns, bench_dir)
    if not tree_data or not direct_data:
        print("  SKIP: missing wgpu_sync or wgpu_direct data")
        return

    plot_ns, tree_ms, direct_ms, tree_de, direct_de = [], [], [], [], []
    for n in ns:
        if n not in tree_data or n not in direct_data:
            continue
        plot_ns.append(n)
        tree_ms.append(total_ms(tree_data[n]).mean())
        direct_ms.append(total_ms(direct_data[n]).mean())
        # Energy drift: last step value
        tree_de.append(abs(tree_data[n]["energy_drift"].iloc[-1]))
        direct_de.append(abs(direct_data[n]["energy_drift"].iloc[-1]))

    plot_ns = np.array(plot_ns)
    fig, ax1 = plt.subplots()

    ln1, = ax1.plot(plot_ns, direct_ms, "o-", color="tab:red",
                    label=r"Direct $O(N^2)$ — time", markersize=5, zorder=3)
    ln2, = ax1.plot(plot_ns, tree_ms, "s-", color="tab:blue",
                    label=r"Tree $O(N \log N)$ — time", markersize=5, zorder=3)

    # Reference lines
    n0 = plot_ns[0]
    d0 = direct_ms[0]
    ax1.plot(plot_ns, d0 * (plot_ns / n0) ** 2, ":", color="tab:red", alpha=0.25)
    t0 = tree_ms[0]
    ax1.plot(plot_ns, t0 * (plot_ns * np.log2(plot_ns)) / (n0 * np.log2(n0)),
             ":", color="tab:blue", alpha=0.25)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (particles)")
    ax1.set_ylabel("ms / step")

    # Right axis — energy drift
    ax2 = ax1.twinx()
    ln3, = ax2.plot(plot_ns, direct_de, "^--", color="tab:red", alpha=0.45,
                    markersize=4, label=r"Direct — $|\Delta E/E_0|$")
    ln4, = ax2.plot(plot_ns, tree_de, "v--", color="tab:blue", alpha=0.45,
                    markersize=4, label=r"Tree — $|\Delta E/E_0|$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|\Delta E / E_0|$", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.grid(False)

    lns = [ln1, ln2, ln3, ln4]
    ax1.legend(lns, [l.get_label() for l in lns], fontsize=7, loc="upper left")
    ax1.set_title("Direct vs Tree: Runtime and Energy Conservation")

    path = os.path.join(fig_dir, "fig_crossover")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.{{pdf,png}}")


def fig3_web_native(fig_dir, bench_dir):
    """Grouped bar chart: native (wgpu) vs browser (Chrome)."""
    ns = [1000, 5000, 10000, 50000, 100000]
    native_data = load_experiment("wgpu_sync", ns, bench_dir)
    chrome_data = load_experiment("chrome_sync", ns, bench_dir)
    if not native_data:
        print("  SKIP: no wgpu_sync data")
        return
    if not chrome_data:
        print("  SKIP: no chrome_sync data (run browser benchmarks first)")
        return

    plot_ns, native_ms, chrome_ms = [], [], []
    for n in ns:
        if n not in native_data or n not in chrome_data:
            continue
        plot_ns.append(n)
        native_ms.append(total_ms(native_data[n]).mean())
        chrome_ms.append(total_ms(chrome_data[n]).mean())

    fig, ax = plt.subplots()
    x = np.arange(len(plot_ns))
    width = 0.35

    ax.bar(x - width / 2, native_ms, width,
           label="Native (wgpu)", color="tab:blue")
    ax.bar(x + width / 2, chrome_ms, width,
           label="Browser (Chrome)", color="tab:orange", hatch="//")

    for i in range(len(plot_ns)):
        overhead = chrome_ms[i] / native_ms[i]
        y_top = max(native_ms[i], chrome_ms[i])
        ax.text(i, y_top * 1.05, f"{overhead:.1f}\u00d7",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in plot_ns], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Native vs Browser Performance")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_web_native")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.{{pdf,png}}")


def fig4_lbvh_breakdown(fig_dir, bench_dir):
    """Stacked bar chart of LBVH per-pass timing."""
    ns = [1000, 5000, 10000, 50000, 100000]
    data = load_experiment("wgpu_lbvh", ns, bench_dir)
    if not data:
        print("  SKIP: no wgpu_lbvh data")
        return

    pass_cols = ["bbox_ms", "morton_ms", "radix_sort_ms",
                 "karras_ms", "leaf_init_ms", "aggregate_ms"]
    labels = ["AABB reduce", "Morton codes", "Radix sort",
              "Karras build", "Leaf init", "Aggregation"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    plot_ns = [n for n in ns if n in data]
    pass_means = []
    for col in pass_cols:
        vals = [data[n][col].mean() for n in plot_ns]
        pass_means.append(vals)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(plot_ns))
    width = 0.6
    bottom = np.zeros(len(plot_ns))

    for vals, label, color in zip(pass_means, labels, colors):
        v = np.array(vals)
        ax.bar(x, v, width, bottom=bottom, label=label, color=color)
        bottom += v

    for i in range(len(plot_ns)):
        dominant_idx = max(range(len(pass_means)),
                          key=lambda j: pass_means[j][i])
        dominant_val = pass_means[dominant_idx][i]
        total_val = bottom[i]
        pct = dominant_val / total_val * 100
        ax.text(i, total_val + 0.15,
                f"{labels[dominant_idx].split()[0]}\n{pct:.0f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in plot_ns], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms")
    ax.set_ylim(top=14)
    ax.set_title("LBVH Construction: Per-Pass Breakdown")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    path = os.path.join(fig_dir, "fig_lbvh_breakdown")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.{{pdf,png}}")


def fig5_cross_backend(fig_dir, bench_dir):
    """Grouped bar chart: Dawn vs wgpu vs Chrome vs Safari."""
    ns = [1000, 10000, 100000]
    backends = {
        "Dawn": ("dawn_sync", ns),
        "wgpu-native": ("wgpu_sync", ns),
        "Chrome": ("chrome_sync", ns),
        "Safari": ("safari_sync", ns),
    }
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:purple"]

    backend_means = {}
    for label, (prefix, exp_ns) in backends.items():
        data = load_experiment(prefix, exp_ns, bench_dir)
        if not data:
            print(f"  Warning: no {prefix} data, skipping {label}")
            continue
        means = []
        for n in ns:
            if n in data:
                means.append(total_ms(data[n]).mean())
            else:
                means.append(0)
        backend_means[label] = means

    if len(backend_means) < 2:
        print("  SKIP: need at least 2 backends for comparison")
        return

    fig, ax = plt.subplots()
    n_backends = len(backend_means)
    width = 0.8 / n_backends
    x = np.arange(len(ns))

    for i, (label, means) in enumerate(backend_means.items()):
        offset = (i - n_backends / 2 + 0.5) * width
        color = colors[i % len(colors)]
        ax.bar(x + offset, means, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=8)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms / step")
    ax.set_title("Cross-Backend Performance Comparison")
    ax.legend(fontsize=8)

    path = os.path.join(fig_dir, "fig_cross_backend")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  Saved: {path}.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════
# Table data extraction (for paper tables)
# ═══════════════════════════════════════════════════════════════════

def print_tables(bench_dir):
    """Print computed table values to stdout for copy-pasting into the paper."""
    ns = [1000, 5000, 10000, 50000, 100000]

    # Table 1: Performance summary
    print("\n" + "=" * 70)
    print("Table 1: @tab:performance-summary (wgpu_sync)")
    print("=" * 70)
    data = load_experiment("wgpu_sync", ns, bench_dir)
    print(f"{'N':>8} {'ms/step':>8} {'std':>6} {'CV':>6} {'tree':>8} {'force':>8} {'integ':>8}")
    for n in ns:
        if n not in data:
            print(f"{n:>8}  MISSING")
            continue
        df = data[n]
        ms = total_ms(df)
        s = compute_stats(ms)
        print(f"{n:>8} {s['mean']:>8.2f} {s['std']:>6.2f} {s['cv']:>6.2f} "
              f"{df['tree_build_ms'].mean():>8.2f} {df['force_ms'].mean():>8.2f} "
              f"{df['integrate_ms'].mean():>8.2f}")

    # Table 2: LBVH breakdown
    print("\n" + "=" * 70)
    print("Table 2: @tab:lbvh-breakdown (wgpu_lbvh)")
    print("=" * 70)
    data = load_experiment("wgpu_lbvh", ns, bench_dir)
    cols = ["bbox_ms", "morton_ms", "radix_sort_ms", "karras_ms", "leaf_init_ms", "aggregate_ms", "force_ms"]
    headers = ["AABB", "Morton", "Sort", "Karras", "Leaf", "Aggr", "Force"]
    print(f"{'N':>8} " + " ".join(f"{h:>8}" for h in headers))
    for n in ns:
        if n not in data:
            print(f"{n:>8}  MISSING")
            continue
        df = data[n]
        vals = " ".join(f"{df[c].mean():>8.2f}" for c in cols)
        print(f"{n:>8} {vals}")

    # Table 3: Crossover
    print("\n" + "=" * 70)
    print("Table 3: @tab:crossover (wgpu_sync + wgpu_direct)")
    print("=" * 70)
    tree = load_experiment("wgpu_sync", ns, bench_dir)
    direct = load_experiment("wgpu_direct", ns, bench_dir)
    print(f"{'N':>8} {'Direct ms':>10} {'Tree ms':>10} {'Direct dE':>12} {'Tree dE':>12}")
    for n in ns:
        d_ms = f"{total_ms(direct[n]).mean():.2f}" if n in direct else "MISSING"
        t_ms = f"{total_ms(tree[n]).mean():.2f}" if n in tree else "MISSING"
        d_de = f"{abs(direct[n]['energy_drift'].iloc[-1]):.2e}" if n in direct else "MISSING"
        t_de = f"{abs(tree[n]['energy_drift'].iloc[-1]):.2e}" if n in tree else "MISSING"
        print(f"{n:>8} {d_ms:>10} {t_ms:>10} {d_de:>12} {t_de:>12}")

    # Table 5: Cross-backend
    backend_ns = [1000, 10000, 100000]
    print("\n" + "=" * 70)
    print("Table 5: @tab:cross-backend")
    print("=" * 70)
    backends = [
        ("Dawn", "dawn_sync"),
        ("wgpu-native", "wgpu_sync"),
        ("Chrome", "chrome_sync"),
        ("Safari", "safari_sync"),
    ]
    print(f"{'Backend':>12} " + " ".join(f"{'N='+str(n):>10}" for n in backend_ns) + f" {'CV(100K)':>10}")
    for label, prefix in backends:
        data = load_experiment(prefix, backend_ns, bench_dir)
        vals = []
        for n in backend_ns:
            if n in data:
                vals.append(f"{total_ms(data[n]).mean():.2f}")
            else:
                vals.append("MISSING")
        cv = ""
        if 100000 in data:
            s = compute_stats(total_ms(data[100000]))
            cv = f"{s['cv']:.2f}"
        else:
            cv = "—"
        print(f"{label:>12} " + " ".join(f"{v:>10}" for v in vals) + f" {cv:>10}")

    # Table 6: Web vs native
    print("\n" + "=" * 70)
    print("Table 6: @tab:web-native")
    print("=" * 70)
    native = load_experiment("wgpu_sync", ns, bench_dir)
    chrome = load_experiment("chrome_sync", ns, bench_dir)
    print(f"{'N':>8} {'Native':>8} {'Chrome':>8} {'Overhead':>10} {'Native dE':>12}")
    for n in ns:
        n_ms = f"{total_ms(native[n]).mean():.2f}" if n in native else "MISSING"
        c_ms = f"{total_ms(chrome[n]).mean():.2f}" if n in chrome else "MISSING"
        if n in native and n in chrome:
            oh = f"{total_ms(chrome[n]).mean() / total_ms(native[n]).mean():.1f}x"
        else:
            oh = "—"
        n_de = f"{abs(native[n]['energy_drift'].iloc[-1]):.2e}" if n in native else "—"
        print(f"{n:>8} {n_ms:>8} {c_ms:>8} {oh:>10} {n_de:>12}")

    # Table 7: Theta sweep
    print("\n" + "=" * 70)
    print("Table 7: @tab:theta-sweep")
    print("=" * 70)
    thetas = [0.3, 0.5, 0.7, 1.0]
    print(f"{'theta':>8} {'ms/step':>8} {'energy_drift':>14}")
    for theta in thetas:
        path = os.path.join(bench_dir, f"theta_{theta}_N5000.csv")
        if os.path.exists(path) and os.path.getsize(path) > 200:
            df = load_csv(path)
            ms = total_ms(df)
            de = abs(df["energy_drift"].iloc[-1])
            print(f"{theta:>8.1f} {ms.mean():>8.2f} {de:>14.2e}")
        else:
            print(f"{theta:>8.1f}  MISSING")

    # Table 8: Two-body validation (Group 1)
    print("\n" + "=" * 70)
    print("Table 8: @tab:twobody-validation (twobody dt sweep)")
    print("=" * 70)
    dts_twobody = [0.0001, 0.0005, 0.001, 0.005]
    print(f"{'dt':>8} {'energy_drift':>14} {'|p|':>14}")
    for dt in dts_twobody:
        path = os.path.join(bench_dir, f"dt_{dt}_N2.csv")
        if os.path.exists(path) and os.path.getsize(path) > 200:
            df = load_csv(path)
            de = abs(df["energy_drift"].iloc[-1])
            pmag = np.sqrt(df["px"].iloc[-1]**2 + df["py"].iloc[-1]**2 + df["pz"].iloc[-1]**2)
            print(f"{dt:>8g} {de:>14.2e} {pmag:>14.2e}")
        else:
            print(f"{dt:>8g}  MISSING")

    # Table 9: Timestep sweep (Group 2c)
    print("\n" + "=" * 70)
    print("Table 9: @tab:dt-sweep (N=5000)")
    print("=" * 70)
    dts = [0.0001, 0.0005, 0.001, 0.005]
    print(f"{'dt':>8} {'ms/step':>8} {'energy_drift':>14}")
    for dt in dts:
        path = os.path.join(bench_dir, f"dt_{dt}_N5000.csv")
        if os.path.exists(path) and os.path.getsize(path) > 200:
            df = load_csv(path)
            ms = total_ms(df)
            de = abs(df["energy_drift"].iloc[-1])
            print(f"{dt:>8g} {ms.mean():>8.2f} {de:>14.2e}")
        else:
            print(f"{dt:>8g}  MISSING")

    # Table 10: Softening sweep (Group 2e)
    print("\n" + "=" * 70)
    print("Table 10: @tab:softening-sweep (N=5000)")
    print("=" * 70)
    softs = [0.1, 0.25, 0.5, 1.0, 2.0]
    print(f"{'epsilon':>8} {'ms/step':>8} {'energy_drift':>14}")
    for soft in softs:
        path = os.path.join(bench_dir, f"softening_{soft}_N5000.csv")
        if os.path.exists(path) and os.path.getsize(path) > 200:
            df = load_csv(path)
            ms = total_ms(df)
            de = abs(df["energy_drift"].iloc[-1])
            print(f"{soft:>8g} {ms.mean():>8.2f} {de:>14.2e}")
        else:
            print(f"{soft:>8g}  MISSING")

    # Table 11: Disk N-scaling (Group 3a)
    print("\n" + "=" * 70)
    print("Table 11: @tab:disk-scaling (disk_sync)")
    print("=" * 70)
    data = load_experiment("disk_sync", ns, bench_dir)
    print(f"{'N':>8} {'ms/step':>8} {'tree':>8} {'force':>8} {'integ':>8}")
    for n in ns:
        if n not in data:
            print(f"{n:>8}  MISSING")
            continue
        df = data[n]
        ms = total_ms(df)
        print(f"{n:>8} {ms.mean():>8.2f} {df['tree_build_ms'].mean():>8.2f} "
              f"{df['force_ms'].mean():>8.2f} {df['integrate_ms'].mean():>8.2f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate figures for capstone paper")
    parser.add_argument("--fig-dir", default=FIG_DIR,
                        help="Output directory for figures")
    parser.add_argument("--bench-dir", default=BENCH_DIR,
                        help="Directory containing benchmark CSVs")
    parser.add_argument("--fig", nargs="*", type=int,
                        help="Figure numbers to generate (default: all)")
    parser.add_argument("--print-tables", action="store_true",
                        help="Print table data to stdout")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    if args.print_tables:
        print_tables(args.bench_dir)
        return

    figs = args.fig or [1, 2, 3, 4, 5]

    if 1 in figs:
        print("\nFigure 1: Plummer N-scaling (log-log)")
        fig1_n_scaling_plummer(args.fig_dir, args.bench_dir)

    if 2 in figs:
        print("\nFigure 2: Direct vs tree crossover (dual-axis)")
        fig2_crossover(args.fig_dir, args.bench_dir)

    if 3 in figs:
        print("\nFigure 3: Native vs browser (wgpu vs Chrome)")
        fig3_web_native(args.fig_dir, args.bench_dir)

    if 4 in figs:
        print("\nFigure 4: LBVH pass breakdown (stacked bar)")
        fig4_lbvh_breakdown(args.fig_dir, args.bench_dir)

    if 5 in figs:
        print("\nFigure 5: Cross-backend comparison")
        fig5_cross_backend(args.fig_dir, args.bench_dir)


if __name__ == "__main__":
    main()
