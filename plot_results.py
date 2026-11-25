"""Generate noninteractive research figures from completed eSIM runs."""
import argparse
from collections import deque
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from run_analysis import ALGORITHMS, collect_results, describe, history


LABELS = {"ql": "Q-learning", "sac": "SAC", "fixed": "Fixed"}
COLORS = {"ql": "#2166ac", "sac": "#23805b", "fixed": "#d97728"}


def chart(title, ylabel, xlabel=None):
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=300)
    plt.close(fig)


def grouped_bars(out, groups, title, ylabel, field_a, field_b, filename):
    fig, ax = chart(title, ylabel)
    x = np.arange(len(groups))
    for i, (algorithm, runs) in enumerate(groups.items()):
        rows = [row for _, _, row in runs]
        for offset, field, color_key, label in (
            (-0.19, field_a, algorithm, "RL provider"), (0.19, field_b, "fixed", "Fixed provider")
        ):
            stats = describe(row[field] for row in rows)
            if stats["mean"] is not None:
                ax.bar(i + offset, stats["mean"], 0.36, yerr=stats["sd"],
                       color=COLORS[color_key], capsize=4, label=label if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] + " vs Fixed" for a in groups])
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    save(fig, out, filename)


def per_run_revenue(out, algorithm, runs):
    fig, ax = chart(LABELS[algorithm] + " vs fixed: testing revenue by run",
                    "Total testing revenue (KZT)", "Independent run")
    rows = [row for _, _, row in runs]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [row["revenue_a"] for row in rows], 0.36,
           color=COLORS[algorithm], label=LABELS[algorithm])
    ax.bar(x + 0.19, [row["revenue_b"] for row in rows], 0.36,
           color=COLORS["fixed"], label=LABELS["fixed"])
    ax.set_xticks(x)
    ax.set_xticklabels(["{:02d}".format(row["run"]) for row in rows])
    ax.legend()
    save(fig, out, algorithm + "_vs_fixed_revenue")


def advantage_chart(out, groups):
    fig, ax = chart("Testing revenue advantage over the competing provider", "Revenue advantage (%)")
    for i, (algorithm, runs) in enumerate(groups.items()):
        stats = describe(row["revenue_advantage_pct"] for _, _, row in runs)
        if stats["mean"] is not None:
            ax.bar(i, stats["mean"], yerr=stats["sd"], color=COLORS[algorithm], capsize=4)
        else:
            ax.annotate("Undefined", (i, 0), xytext=(0, 6), textcoords="offset points", ha="center")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([LABELS[a] for a in groups])
    save(fig, out, "revenue_advantage")


def plot_band(ax, x, series, label, color):
    values = np.asarray(series)
    center = np.mean(values, axis=0)
    ax.plot(x, center, label=label, color=color)
    if len(values) > 1:
        sd = np.std(values, axis=0, ddof=1)
        ax.fill_between(x, center - sd, center + sd, alpha=0.18, color=color)


def evaluation_curves(out, groups, price_steps, price_start):
    price_fig, price_ax = chart("Evaluation pricing trajectories", "Package price (KZT)", "Testing step")
    fixed_prices = set()
    for algorithm, runs in groups.items():
        length = min(conf["eval_timesteps"] for _, conf, _ in runs)
        totals_a, totals_b = [], []
        for i, (folder, conf, _) in enumerate(runs):
            prices = []
            revenues_a = np.empty(length)
            revenues_b = np.empty(length)
            for row in history(folder / "eval_history.jsonl", conf["eval_timesteps"]):
                step = row["step"]
                if step <= length:
                    revenues_a[step-1], revenues_b[step-1] = row["revenue_a"], row["revenue_b"]
                if price_start <= step < price_start + price_steps:
                    prices.append(row["price_a"])
                    fixed_prices.add(row["price_b"])
            totals_a.append(np.cumsum(revenues_a))
            totals_b.append(np.cumsum(revenues_b))
            price_ax.plot(range(price_start, price_start + len(prices)), prices, color=COLORS[algorithm],
                          alpha=0.65, label=LABELS[algorithm] if i == 0 and prices else None)

        x = np.arange(1, length + 1)
        fig, ax = chart(LABELS[algorithm] + " vs fixed: cumulative testing revenue",
                        "Cumulative revenue (KZT)", "Testing step")
        plot_band(ax, x, totals_a, LABELS[algorithm], COLORS[algorithm])
        plot_band(ax, x, totals_b, LABELS["fixed"], COLORS["fixed"])
        ax.legend()
        save(fig, out, algorithm + "_cumulative_revenue")

    for price in sorted(fixed_prices):
        price_ax.axhline(price, color=COLORS["fixed"], linestyle="--", label="Fixed ({:,.0f} KZT)".format(price))
    if fixed_prices:
        price_ax.legend()
    else:
        price_ax.text(0.5, 0.5, "No steps in the requested range", transform=price_ax.transAxes, ha="center")
    save(price_fig, out, "price_trajectories")


def moving_revenue(path, total_steps, window, sample_steps):
    recent = deque()
    total = 0.0
    means = []
    steps = iter(sample_steps)
    next_step = next(steps, None)
    for row in history(path, total_steps):
        revenue = row["revenue_a"]
        if len(recent) == window:
            total -= recent.popleft()
        recent.append(revenue)
        total += revenue
        if row["step"] == next_step:
            means.append(total / window)
            next_step = next(steps, None)
    return means


def convergence_chart(out, groups, window):
    fig, ax = chart("Training revenue ({}-step moving average)".format(window),
                    "Revenue per period (KZT)", "Training step")
    for algorithm, runs in groups.items():
        available = [(folder / "train_history.jsonl", conf["train_timesteps"])
                     for folder, conf, _ in runs if (folder / "train_history.jsonl").is_file()]
        if not available:
            continue
        length = min(steps for _, steps in available)
        if length < window:
            continue
        sample_steps = np.unique(np.linspace(window, length, min(length-window+1, 2000), dtype=int))
        curves = []
        for path, steps in available:
            try:
                curves.append(moving_revenue(path, steps, window, sample_steps))
            except (ValueError, KeyError) as error:
                print("Skipping {}: {}".format(path, error), file=sys.stderr)
        if curves:
            plot_band(ax, sample_steps, curves, LABELS[algorithm], COLORS[algorithm])
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No training histories reach the smoothing window",
                transform=ax.transAxes, ha="center")
    save(fig, out, "learning_convergence")


def boxplot(out, groups):
    fig, ax = chart("Testing revenue variation across independent runs", "Total testing revenue (KZT)")
    values, labels, colors = [], [], []
    for algorithm, runs in groups.items():
        values.extend([[row[field] for _, _, row in runs] for field in ("revenue_a", "revenue_b")])
        labels.extend([LABELS[algorithm], "Fixed vs\n" + LABELS[algorithm]])
        colors.extend([COLORS[algorithm], COLORS["fixed"]])
    artists = ax.boxplot(values, labels=labels, patch_artist=True)
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    save(fig, out, "revenue_boxplot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/esim"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--price-start", type=int, default=1)
    parser.add_argument("--price-steps", type=int, default=250)
    parser.add_argument("--smoothing-window", type=int, default=1000)
    args = parser.parse_args()
    if min(args.price_start, args.price_steps, args.smoothing_window) < 1:
        parser.error("Price range and smoothing window must be positive.")
    grouped = {algorithm: [] for algorithm in ALGORITHMS}
    for result in collect_results(args.input_dir):
        grouped[result[2]["algorithm"]].append(result)
    groups = {algorithm: runs for algorithm, runs in grouped.items() if runs}
    if not groups:
        parser.error("No completed evaluation runs found in {}".format(args.input_dir))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grouped_bars(out, groups, "Mean total testing revenue by matchup", "Total testing revenue (KZT)",
                 "revenue_a", "revenue_b", "revenue_comparison")
    for algorithm, runs in groups.items():
        per_run_revenue(out, algorithm, runs)
    advantage_chart(out, groups)
    evaluation_curves(out, groups, args.price_steps, args.price_start)
    convergence_chart(out, groups, args.smoothing_window)
    boxplot(out, groups)
    grouped_bars(out, groups, "Mean testing market share by matchup", "Share of completed sales (%)",
                 "market_share_a_pct", "market_share_b_pct", "market_share")
    print("Wrote figures to {}".format(out))


if __name__ == "__main__":
    main()
