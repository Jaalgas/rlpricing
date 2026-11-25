"""Summarize actual testing outcomes from completed eSIM runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
import sys

from scipy.stats import t

from strategies import COMPETITORS


ALGORITHMS = ("ql", "sac")
MATCHUPS = tuple((algorithm, competitor) for competitor in COMPETITORS for algorithm in ALGORITHMS)
METRICS = ("revenue_a", "revenue_b", "revenue_difference", "revenue_advantage_pct",
           "sales_a", "sales_b", "no_purchase", "total_customers",
           "market_share_a_pct", "market_share_b_pct", "conversion_rate_pct",
           "average_price_a", "average_price_b", "win")
TOTALS = ("revenue_a", "revenue_b", "sales_a", "sales_b", "no_purchase", "total_customers")


def matchup_name(algorithm, competitor):
    return algorithm + "_vs_" + competitor


def completed_runs(root):
    for algorithm, competitor in MATCHUPS:
        for folder in sorted((root / matchup_name(algorithm, competitor)).glob("run_*")):
            conf = folder / "conf.json"
            if not conf.is_file():
                continue
            config = json.loads(conf.read_text(encoding="utf-8"))
            if config.get("status") == "completed":
                if not (folder / "eval_history.jsonl").is_file():
                    print("Skipping {}: evaluation history is missing.".format(folder), file=sys.stderr)
                    continue
                if (config["algorithm"], config["gym_params"]["competitor"]) != (algorithm, competitor):
                    print("Skipping {}: matchup does not match its configuration.".format(folder),
                          file=sys.stderr)
                    continue
                yield folder, config


def history(path, expected_steps):
    count = 0
    with path.open(encoding="utf-8") as source:
        for count, line in enumerate(source, 1):
            row = json.loads(line)
            if row["step"] != count:
                raise ValueError("{} has nonconsecutive steps at line {}".format(path, count))
            yield row
    if count != expected_steps:
        raise ValueError("{} has {} rows, expected {}".format(path, count, expected_steps))


def ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if denominator else None


def evaluation_metrics(rows, config):
    totals = dict.fromkeys(TOTALS, 0)
    for row in rows:
        for metric in totals:
            totals[metric] += row[metric]
    a, b = totals["revenue_a"], totals["revenue_b"]
    sold = totals["sales_a"] + totals["sales_b"]
    return dict(
        algorithm=config["algorithm"], competitor=config["gym_params"]["competitor"],
        reward_mode=config["gym_params"].get("reward_mode", "own_revenue"),
        revenue_weight=config["gym_params"].get("revenue_weight"),
        competitive_weight=config["gym_params"].get("competitive_weight"),
        run=config["run"], evaluation_seed=config["evaluation_seed"],
        eval_steps=config["eval_timesteps"], **totals, revenue_difference=a-b,
        revenue_advantage_pct=ratio(a-b, b, 100),
        market_share_a_pct=ratio(totals["sales_a"], sold, 100),
        market_share_b_pct=ratio(totals["sales_b"], sold, 100),
        conversion_rate_pct=ratio(sold, totals["total_customers"], 100),
        average_price_a=ratio(a, totals["sales_a"]),
        average_price_b=ratio(b, totals["sales_b"]), win=int(a > b))


def collect_results(root):
    for folder, config in completed_runs(root):
        try:
            metrics = evaluation_metrics(
                history(folder / "eval_history.jsonl", config["eval_timesteps"]), config)
        except (ValueError, KeyError) as error:
            print("Skipping {}: {}".format(folder, error), file=sys.stderr)
            continue
        yield folder, config, metrics


def describe(values):
    values = [value for value in values if value is not None]
    n = len(values)
    if not n:
        return dict(n=0, mean=None, sd=None, median=None, ci95_low=None, ci95_high=None)
    center = mean(values)
    spread = stdev(values) if n > 1 else None
    margin = t.ppf(0.975, n-1) * spread / n**0.5 if spread is not None else None
    return dict(n=n, mean=center, sd=spread, median=median(values),
                ci95_low=center-margin if margin is not None else None,
                ci95_high=center+margin if margin is not None else None)


def pair_key(config):
    market = dict(config["gym_params"])
    market.pop("mode")
    if market.get("keep_history") is False:
        del market["keep_history"]
    return (config["run"], config["evaluation_seed"], config["eval_timesteps"],
            config["package_gb"], json.dumps(market, sort_keys=True))


def summarize(results):
    for algorithm, competitor in MATCHUPS:
        runs = [row for _, _, row in results
                if (row["algorithm"], row["competitor"]) == (algorithm, competitor)]
        if not runs:
            continue
        comparison = matchup_name(algorithm, competitor)
        for metric in METRICS:
            if metric == "win":
                wins = [row["win"] for row in runs]
                yield dict(comparison=comparison, metric="win_rate_pct",
                           n=len(wins), mean=100 * mean(wins))
            else:
                yield dict(comparison=comparison, metric=metric, **describe(row[metric] for row in runs))

    for competitor in COMPETITORS:
        by_algorithm = {
            algorithm: {pair_key(conf): row for _, conf, row in results
                        if (row["algorithm"], row["competitor"]) == (algorithm, competitor)}
            for algorithm in ALGORITHMS
        }
        ql, sac = by_algorithm["ql"], by_algorithm["sac"]
        pairs = [(sac[key], ql[key]) for key in sorted(sac.keys() & ql.keys())]
        if pairs:
            for metric in ("revenue_a", "revenue_difference", "revenue_advantage_pct"):
                differences = (left[metric] - right[metric] for left, right in pairs
                               if left[metric] is not None and right[metric] is not None)
                yield dict(comparison="matched_sac_minus_ql_vs_" + competitor, metric=metric,
                           **describe(differences))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/esim"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = list(collect_results(args.input_dir))
    if not results:
        parser.error("No completed evaluation runs found in {}".format(args.input_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "individual_runs.csv",
              ("algorithm", "competitor", "reward_mode", "revenue_weight",
               "competitive_weight", "run", "evaluation_seed", "eval_steps") + METRICS,
              (row for _, _, row in results))
    write_csv(args.output_dir / "evaluation_summary.csv",
              ("comparison", "metric", "n", "mean", "sd", "median", "ci95_low", "ci95_high"),
              summarize(results))
    print("Summarized {} completed evaluation runs in {}".format(len(results), args.output_dir))


if __name__ == "__main__":
    main()
