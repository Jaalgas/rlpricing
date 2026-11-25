"""Evaluate frozen Q-learning and SAC policies head-to-head."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from stable_baselines import SAC

from environments.duopoly import MarketEnv
from gymql.strategy import QLModel
from simtools import price_grid


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ROOT / "experiments" / "revenue_30_own_70_comp_50k_10k_one_seed"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenQL:
    """Load a Q-learning checkpoint and expose greedy actions without learning."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.config = read_json(self.run_dir / "conf.json")
        if self.config.get("status") != "completed":
            raise ValueError("Q-learning checkpoint run is not completed: {}".format(run_dir))
        if self.config["algorithm"] != "ql":
            raise ValueError("Expected a Q-learning run at {}".format(run_dir))

        env_params = dict(self.config["gym_params"])
        env_params["sim_params"] = dict(env_params["sim_params"], seed=self.config["evaluation_seed"])
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default(), self.session.as_default():
            self.env = MarketEnv(**env_params)
            self.model = QLModel(self.env, **self.config["model_params"])
            self.model.load_weights(str(self.run_dir / "trained_model.weights.h5"))

    def action(self, observation):
        with self.graph.as_default(), self.session.as_default():
            return int(self.model.predict(observation)[0][0])

    def close(self):
        self.session.close()


def load_sac(run_dir):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "conf.json")
    if config.get("status") != "completed":
        raise ValueError("SAC checkpoint run is not completed: {}".format(run_dir))
    if config["algorithm"] != "sac":
        raise ValueError("Expected a SAC run at {}".format(run_dir))
    checkpoint = run_dir / "trained_model.zip"
    return config, SAC.load(str(checkpoint), env=None)


def normalized_observation(observation, market):
    scale = market["max_p"] - market["min_p"]
    return (np.asarray(observation, dtype=np.float32) - market["min_p"]) / scale - 0.5


def choose_action(algorithm, model, observation, market):
    if algorithm == "ql":
        return model.action(observation)
    action, _ = model.predict(normalized_observation(observation, market)[None, :],
                              deterministic=True)
    return np.asarray(action).reshape(-1)[0]


def choose_price(algorithm, model, observation, market):
    if algorithm == "ql":
        index = int(model.action(observation))
        prices = price_grid(market["min_p"], market["max_p"], market["step_size"])
        return float(prices[index])
    action = float(choose_action("sac", model, observation, market))
    price = round(market["min_p"] + (market["max_p"] - market["min_p"]) / 2 + action, 2)
    return float(np.clip(price, market["min_p"], market["max_p"]))


def evaluate_orientation(algorithm_a, algorithm_b, ql, sac, market_template,
                         output_dir, seed, steps, weights):
    market = dict(market_template)
    market["mode"] = "discrete" if algorithm_a == "ql" else "continuous"
    market["sim_params"] = dict(market["sim_params"], seed=seed)
    market["reward_mode"] = "revenue_mix"
    market["revenue_weight"], market["competitive_weight"] = weights
    env = MarketEnv(**market)
    env.reset(seed=seed)

    policies = {"ql": ql, "sac": sac}
    policy_a = policies[algorithm_a]
    policy_b = policies[algorithm_b]
    result_path = output_dir / "eval_history.jsonl"
    totals = {key: 0 for key in ("sales_a", "sales_b", "no_purchase", "total_customers",
                                 "reward_a", "reward_b")}

    with result_path.open("w", encoding="utf-8") as history:
        for step in range(steps):
            observation_a = env.observe()
            action_a = choose_action(algorithm_a, policy_a, observation_a, market)

            def provider_b_policy(observation_b):
                return choose_price(algorithm_b, policy_b, observation_b, market)

            _, _, _, info = env.step(action_a, provider_b_policy=provider_b_policy)
            info.update(algorithm_a=algorithm_a, algorithm_b=algorithm_b,
                        revenue_weight=weights[0], competitive_weight=weights[1])
            info["step"] = step + 1
            totals["sales_a"] += info["sales_a"]
            totals["sales_b"] += info["sales_b"]
            totals["no_purchase"] += info["no_purchase"]
            totals["total_customers"] += info["total_customers"]
            totals["reward_a"] += info["reward_a"]
            totals["reward_b"] += info["reward_b"]
            history.write(json.dumps(info, separators=(",", ":"), allow_nan=False) + "\n")

    revenue_a = env.total_revenue_a
    revenue_b = env.total_revenue_b
    summary = dict(
        algorithm_a=algorithm_a,
        algorithm_b=algorithm_b,
        evaluation_seed=seed,
        eval_steps=steps,
        revenue_weight=weights[0],
        competitive_weight=weights[1],
        revenue_a=revenue_a,
        revenue_b=revenue_b,
        revenue_difference_a_minus_b=revenue_a - revenue_b,
        revenue_advantage_pct=(100 * (revenue_a - revenue_b) / revenue_b
                               if revenue_b else None),
        win_a=revenue_a > revenue_b,
        **totals
    )
    (output_dir / "conf.json").write_text(
        json.dumps(dict(summary, learning_performed=False), indent=2), encoding="utf-8")
    print("{} vs {}: A {:.2f}, B {:.2f} KZT; learning disabled".format(
        algorithm_a.upper(), algorithm_b.upper(), revenue_a, revenue_b), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ql-run", type=Path,
                        default=DEFAULT_RUNS / "ql_vs_fixed" / "run_00")
    parser.add_argument("--sac-run", type=Path,
                        default=DEFAULT_RUNS / "sac_vs_fixed" / "run_00")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "ql_sac_duel_30_own_70")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Deterministic evaluation periods per seating order.")
    parser.add_argument("--seed", type=int,
                        help="Customer seed; defaults to the checkpoint evaluation seed.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir.")

    ql_config = read_json(args.ql_run / "conf.json")
    sac_config = read_json(args.sac_run / "conf.json")
    weights = (ql_config["gym_params"].get("revenue_weight"),
               ql_config["gym_params"].get("competitive_weight"))
    if weights != (0.3, 0.7):
        parser.error("The default duel uses checkpoints trained with 0.3 revenue / 0.7 competitive weights.")
    if (sac_config["gym_params"].get("revenue_weight"),
            sac_config["gym_params"].get("competitive_weight")) != weights:
        parser.error("QL and SAC checkpoints must use the same reward weights.")
    if ql_config["evaluation_seed"] != sac_config["evaluation_seed"]:
        parser.error("QL and SAC checkpoint runs must have matched evaluation seeds.")

    seed = args.seed if args.seed is not None else ql_config["evaluation_seed"]
    market_template = dict(ql_config["gym_params"])
    market_template["sim_params"] = dict(market_template["sim_params"])
    market_template["competitor"] = "fixed"
    market_template["comp_params"] = ql_config["gym_params"].get("comp_params", {"p": 4000})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        mode="frozen_q_learning_vs_sac",
        learning_performed=False,
        eval_steps=args.steps,
        evaluation_seed=seed,
        revenue_weight=weights[0],
        competitive_weight=weights[1],
        ql_checkpoint=str((args.ql_run / "trained_model.weights.h5").resolve()),
        sac_checkpoint=str((args.sac_run / "trained_model.zip").resolve()),
        seating_orders=["ql_vs_sac", "sac_vs_ql"],
    )
    (args.output_dir / "duel_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    ql = FrozenQL(args.ql_run)
    sac_conf, sac = load_sac(args.sac_run)
    rows = []
    try:
        for algorithm_a, algorithm_b in (("ql", "sac"), ("sac", "ql")):
            folder = args.output_dir / (algorithm_a + "_vs_" + algorithm_b)
            folder.mkdir(parents=True, exist_ok=True)
            rows.append(evaluate_orientation(
                algorithm_a, algorithm_b, ql, sac, market_template, folder,
                seed, args.steps, weights))
    finally:
        ql.close()
        sac.sess.close()

    with (args.output_dir / "duel_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
