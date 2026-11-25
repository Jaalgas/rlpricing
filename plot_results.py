"""Generate noninteractive research figures from completed eSIM runs."""
import argparse
from collections import deque
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from run_analysis import MATCHUPS, collect_results, describe, history, matchup_name


LABELS = {"ql": "Q-learning", "sac": "SAC", "fixed": "Fixed",
          "twobound": "TwoBound", "cartel": "CartelTwoBound"}
COLORS = {"ql": "#2166ac", "sac": "#23805b", "fixed": "#d97728",
          "twobound": "#9c4996", "cartel": "#a75241"}
LINE_STYLES = {"fixed": "-", "twobound": "--", "cartel": ":"}


def matchup_label(algorithm, competitor, newline=False):
    separator = "\nvs " if newline else " vs "
    return LABELS[algorithm] + separator + LABELS[competitor]


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
    fig.set_figwidth(max(8, 1.7 * len(groups)))
    x = np.arange(len(groups))
    for i, ((algorithm, competitor), runs) in enumerate(groups.items()):
        rows = [row for _, _, row in runs]
        for offset, field, color_key in ((-0.19, field_a, algorithm), (0.19, field_b, competitor)):
            stats = describe(row[field] for row in rows)
            if stats["mean"] is not None:
                ax.bar(i + offset, stats["mean"], 0.36, yerr=stats["sd"],
                       color=COLORS[color_key], capsize=4)
    ax.set_xlabel("Left bar: learner. Right bar: competitor.")
    ax.set_xticks(x)
    ax.set_xticklabels([matchup_label(*key, newline=True) for key in groups])
    save(fig, out, filename)


def per_run_revenue(out, algorithm, competitor, runs):
    fig, ax = chart(matchup_label(algorithm, competitor) + ": testing revenue by run",
                    "Total testing revenue (KZT)", "Independent run")
    rows = [row for _, _, row in runs]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [row["revenue_a"] for row in rows], 0.36,
           color=COLORS[algorithm], label=LABELS[algorithm])
    ax.bar(x + 0.19, [row["revenue_b"] for row in rows], 0.36,
           color=COLORS[competitor], label=LABELS[competitor])
    ax.set_xticks(x)
    ax.set_xticklabels(["{:02d}".format(row["run"]) for row in rows])
    ax.legend()
    save(fig, out, matchup_name(algorithm, competitor) + "_revenue")


def advantage_chart(out, groups):
    fig, ax = chart("Testing revenue advantage over the competing provider", "Revenue advantage (%)")
    fig.set_figwidth(max(8, 1.7 * len(groups)))
    for i, ((algorithm, competitor), runs) in enumerate(groups.items()):
        stats = describe(row["revenue_advantage_pct"] for _, _, row in runs)
        if stats["mean"] is not None:
            ax.bar(i, stats["mean"], yerr=stats["sd"], color=COLORS[algorithm], capsize=4)
        else:
            ax.annotate("Undefined", (i, 0), xytext=(0, 6), textcoords="offset points", ha="center")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([matchup_label(*key, newline=True) for key in groups])
    save(fig, out, "revenue_advantage")


def plot_band(ax, x, series, label, color, linestyle="-"):
    values = np.asarray(series)
    center = np.mean(values, axis=0)
    ax.plot(x, center, label=label, color=color, linestyle=linestyle)
    if len(values) > 1:
        sd = np.std(values, axis=0, ddof=1)
        ax.fill_between(x, center - sd, center + sd, alpha=0.18, color=color)


def evaluation_curves(out, groups, price_steps, price_start):
    columns = min(2, len(groups))
    rows = (len(groups) + columns - 1) // columns
    price_fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.3 * rows),
                                  squeeze=False, constrained_layout=True)
    price_fig.suptitle("Evaluation pricing trajectories")
    for ax, ((algorithm, competitor), runs) in zip(axes.flat, groups.items()):
        ax.set(title=matchup_label(algorithm, competitor), ylabel="Package price (KZT)",
               xlabel="Testing step")
        length = min(conf["eval_timesteps"] for _, conf, _ in runs)
        totals_a, totals_b = [], []
        for folder, conf, _ in runs:
            times, prices_a, prices_b = [], [], []
            revenues_a = np.empty(length)
            revenues_b = np.empty(length)
            for row in history(folder / "eval_history.jsonl", conf["eval_timesteps"]):
                step = row["step"]
                if step <= length:
                    revenues_a[step-1], revenues_b[step-1] = row["revenue_a"], row["revenue_b"]
                if price_start <= step < price_start + price_steps:
                    times.extend((step, step + conf["gym_params"]["h"]))
                    prices_a.extend((row["price_a"], row["price_a"]))
                    prices_b.extend((row["first_half"].get("price_b", row["price_b"]),
                                     row["second_half"].get("price_b", row["price_b"])))
            totals_a.append(np.cumsum(revenues_a))
            totals_b.append(np.cumsum(revenues_b))
            if times:
                show_labels = not ax.lines
                times.append(min(price_start + price_steps - 1, conf["eval_timesteps"]) + 1)
                ax.step(times, prices_a + prices_a[-1:], where="post", color=COLORS[algorithm],
                        alpha=0.65, label=LABELS[algorithm] if show_labels else None)
                ax.step(times, prices_b + prices_b[-1:], where="post", color=COLORS[competitor],
                        alpha=0.65, label=LABELS[competitor] if show_labels else None)

        x = np.arange(1, length + 1)
        fig, revenue_ax = chart(matchup_label(algorithm, competitor) + ": cumulative testing revenue",
                                "Cumulative revenue (KZT)", "Testing step")
        plot_band(revenue_ax, x, totals_a, LABELS[algorithm], COLORS[algorithm])
        plot_band(revenue_ax, x, totals_b, LABELS[competitor], COLORS[competitor])
        revenue_ax.legend()
        filename = algorithm if competitor == "fixed" else matchup_name(algorithm, competitor)
        save(fig, out, filename + "_cumulative_revenue")
        if ax.lines:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No steps in the requested range", transform=ax.transAxes, ha="center")

    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
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
    for (algorithm, competitor), runs in groups.items():
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
            plot_band(ax, sample_steps, curves, matchup_label(algorithm, competitor),
                      COLORS[algorithm], LINE_STYLES[competitor])
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No training histories reach the smoothing window",
                transform=ax.transAxes, ha="center")
    save(fig, out, "learning_convergence")


def boxplot(out, groups):
    fig, ax = chart("Testing revenue variation across independent runs", "Total testing revenue (KZT)")
    fig.set_figwidth(max(8, 2.5 * len(groups)))
    values, labels, colors = [], [], []
    for (algorithm, competitor), runs in groups.items():
        values.extend([[row[field] for _, _, row in runs] for field in ("revenue_a", "revenue_b")])
        labels.extend([matchup_label(algorithm, competitor, newline=True),
                       matchup_label(competitor, algorithm, newline=True)])
        colors.extend([COLORS[algorithm], COLORS[competitor]])
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
    grouped = {matchup: [] for matchup in MATCHUPS}
    for result in collect_results(args.input_dir):
        row = result[2]
        grouped[(row["algorithm"], row["competitor"])].append(result)
    groups = {matchup: runs for matchup, runs in grouped.items() if runs}
    if not groups:
        parser.error("No completed evaluation runs found in {}".format(args.input_dir))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grouped_bars(out, groups, "Mean total testing revenue by matchup", "Total testing revenue (KZT)",
                 "revenue_a", "revenue_b", "revenue_comparison")
    for (algorithm, competitor), runs in groups.items():
        per_run_revenue(out, algorithm, competitor, runs)
    advantage_chart(out, groups)
    evaluation_curves(out, groups, args.price_steps, args.price_start)
    convergence_chart(out, groups, args.smoothing_window)
    boxplot(out, groups)
    grouped_bars(out, groups, "Mean testing market share by matchup", "Share of completed sales (%)",
                 "market_share_a_pct", "market_share_b_pct", "market_share")
    print("Wrote figures to {}".format(out))


if __name__ == "__main__":
    main()
