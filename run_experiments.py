"""Run Q-learning and SAC against scripted eSIM competitors."""
import argparse
import json
from pathlib import Path
import random

import gym
import numpy as np
import tensorflow as tf
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac import MlpPolicy

from environments.duopoly import MarketEnv
from gymql.strategy import QLModel
from simtools import update
from strategies import COMPETITORS


ROOT = Path(__file__).resolve().parent
PRESETS = {"ql": "gym_ql_nice.json", "sac": "gym_sac.json"}
MODELS = {"ql": "trained_model.weights.h5", "sac": "trained_model.zip"}


class RecordedMarket(gym.Wrapper):
    def __init__(self, env, history, reward_scale=1, normalize=False):
        super().__init__(env)
        self.history = history
        self.reward_scale = reward_scale
        self.normalize = normalize
        if normalize:
            self.observation_space = gym.spaces.Box(-0.5, 0.5, shape=env.observation_space.shape,
                                                    dtype=np.float32)

    def observation(self, obs):
        if self.normalize:
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            return (obs - low) / (high - low) - 0.5
        return obs

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, record = self.env.step(action)
        self.history.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        return self.observation(obs), reward / self.reward_scale, done, record


def read_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_config(spec, algorithm, competitor, index, smoke):
    params = read_config(ROOT / PRESETS[algorithm])["model_params"]
    update(params, spec["model_params"])
    if algorithm == "ql":
        params["store_params"] = {"size": spec["ql_replay_capacity"]}
        if smoke:
            params["store_params"]["size"] = 16
            params["sample_size"] = 33  # Exercise a final minibatch of one transition.
            params["training_eps"] = 16
        elif spec["train_timesteps"] <= spec["ql_replay_capacity"]:
            # A shortened run needs time to update after filling its replay buffer.
            params["store_params"]["size"] = max(1, spec["train_timesteps"] // 2)
    # Match customer seeds across algorithms. Model seeds are independent of both phases.
    seed_states = np.random.SeedSequence([spec["seed"], index]).generate_state(3)
    model_seed, training_seed, evaluation_seed = map(int, seed_states)
    market = dict(spec["gym_params"], mode="discrete" if algorithm == "ql" else "continuous",
                  competitor=competitor, comp_params=spec["competitors"][competitor])
    return dict(algorithm=algorithm, run=index, package_gb=spec["package_gb"],
                gym_params=market, model_params=params, model_seed=model_seed,
                training_seed=training_seed, evaluation_seed=evaluation_seed,
                train_timesteps=spec["train_timesteps"], eval_timesteps=spec["eval_timesteps"],
                sac_reward_scale=spec["sac_reward_scale"] if algorithm == "sac" else 1,
                smoke_test=smoke)


def save_status(path, config, status):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(config, status=status), indent=2), encoding="utf-8")
    temp.replace(path)


def execute_run(folder, config):
    algorithm = config["algorithm"]
    artifact = folder / MODELS[algorithm]
    conf = folder / "conf.json"
    train_history = folder / "train_history.jsonl"
    eval_history = folder / "eval_history.jsonl"
    folder.mkdir(parents=True, exist_ok=True)
    if conf.exists():
        previous = read_config(conf)
        status = previous.pop("status")
        if previous["gym_params"].get("keep_history") is False:
            del previous["gym_params"]["keep_history"]
        if previous != config:
            raise ValueError("Configuration differs for {}. Use a different --output-dir.".format(folder))
        if status == "completed":
            missing = [p.name for p in (artifact, train_history, eval_history) if not p.is_file()]
            if missing:
                raise ValueError("Completed run {} is missing {}".format(folder, ", ".join(missing)))
            print("Skipping completed {}".format(folder), flush=True)
            return
    save_status(conf, config, "running")
    if algorithm == "ql":
        print("Q-learning replay capacity: {}".format(config["model_params"]["store_params"]["size"]), flush=True)
    market_params = dict(config["gym_params"])
    market_params["sim_params"] = dict(market_params["sim_params"], seed=config["training_seed"])
    seed = config["model_seed"]
    sac = algorithm == "sac"
    try:
        env = MarketEnv(**market_params)
        if env.sim.cap != config["package_gb"]:
            raise ValueError("The configured package must match the customer model capacity")
        with tf.Graph().as_default(), tf.Session():
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.action_space.seed(seed)
            with train_history.open("w", encoding="utf-8") as output:
                train_env = DummyVecEnv([lambda: RecordedMarket(
                    env, output, config["sac_reward_scale"], normalize=sac)])
                if sac:
                    model = SAC(MlpPolicy, train_env, seed=seed, **config["model_params"])
                    if config["smoke_test"]:
                        initial_weights = model.get_parameters()
                    model.learn(total_timesteps=config["train_timesteps"])
                    if config["smoke_test"] and all(
                        np.array_equal(value, initial_weights[key])
                        for key, value in model.get_parameters().items()
                    ):
                        raise RuntimeError("SAC smoke test did not update its network")
                else:
                    model = QLModel(train_env, **config["model_params"])
                    model.learn(total_timesteps=config["train_timesteps"])
                    if config["smoke_test"] and not model.q.losses:
                        raise RuntimeError("Q-learning smoke test did not update its network")
                model.save(str(artifact))

            with eval_history.open("w", encoding="utf-8") as output:
                test_env = RecordedMarket(env, output, normalize=sac)
                obs = test_env.reset(seed=config["evaluation_seed"])
                for _ in range(config["eval_timesteps"]):
                    action, _ = model.predict(obs[None, :], deterministic=True) if sac else model.predict(obs)
                    obs, _, _, _ = test_env.step(action)
    except BaseException:
        save_status(conf, config, "incomplete")
        raise
    save_status(conf, config, "completed")
    print("Completed {} (evaluation revenue A {:.2f}, B {:.2f} KZT)".format(
        folder, env.total_revenue_a, env.total_revenue_b), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "gym_esim.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reward", choices=("revenue_mix", "own_revenue", "revenue_lead"),
                        default="revenue_mix",
                        help="Training objective. The default blends 70%% own revenue with 30%% revenue lead.")
    parser.add_argument("--revenue-weight", type=float,
                        help="Own-revenue weight for revenue_mix; weights must sum to one.")
    parser.add_argument("--competitive-weight", type=float,
                        help="Revenue-lead weight for revenue_mix; weights must sum to one.")
    parser.add_argument("--algorithm", choices=tuple(PRESETS))
    parser.add_argument("--competitor", choices=tuple(COMPETITORS))
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    spec = read_config(args.config)
    market = spec["gym_params"]
    market["reward_mode"] = args.reward
    if args.reward == "revenue_mix":
        revenue_weight = args.revenue_weight
        if revenue_weight is None:
            revenue_weight = market.get("revenue_weight", 0.7)
        competitive_weight = args.competitive_weight
        if competitive_weight is None:
            competitive_weight = market.get("competitive_weight", 0.3)
        if (revenue_weight < 0 or competitive_weight < 0
                or not np.isclose(revenue_weight + competitive_weight, 1.0)):
            parser.error("Revenue and competitive weights must be nonnegative and sum to one.")
        market["revenue_weight"] = revenue_weight
        market["competitive_weight"] = competitive_weight
    else:
        if args.revenue_weight is not None or args.competitive_weight is not None:
            parser.error("Set reward weights only when --reward revenue_mix is selected.")
        market.pop("revenue_weight", None)
        market.pop("competitive_weight", None)
    if "competitors" not in spec:
        spec["competitors"] = {market.pop("competitor"): market.pop("comp_params")}
    opponents = spec["competitors"]
    if not opponents or opponents.keys() - COMPETITORS.keys():
        parser.error("Configure one or more of: {}".format(", ".join(COMPETITORS)))
    if args.competitor and args.competitor not in opponents:
        parser.error("The selected competitor is missing from the configuration.")
    if not spec["algorithms"] or any(algorithm not in PRESETS for algorithm in spec["algorithms"]):
        parser.error("The only supported algorithms are Q-learning and SAC.")
    if args.algorithm and args.algorithm not in spec["algorithms"]:
        parser.error("The selected algorithm is missing from the configuration.")
    if args.smoke_test:
        spec.update(train_timesteps=160, eval_timesteps=20, num_experiments=1)
    for flag, key in (("train_steps", "train_timesteps"), ("eval_steps", "eval_timesteps"),
                      ("runs", "num_experiments")):
        value = getattr(args, flag)
        if value is not None:
            spec[key] = value
        if spec[key] < 1:
            parser.error("{} must be positive".format(key))
    if spec["ql_replay_capacity"] < 1 or spec["sac_reward_scale"] <= 0:
        parser.error("Replay capacity and reward scale must be positive.")
    default_folder = "esim_smoke" if args.smoke_test else "esim"
    if market.get("reward_mode") == "revenue_mix":
        default_folder += "_revenue_mix"
    elif market.get("reward_mode") == "revenue_lead":
        default_folder += "_revenue_lead"
    root = args.output_dir or Path("experiments") / default_folder
    algorithms = [args.algorithm] if args.algorithm else spec["algorithms"]
    competitors = [args.competitor] if args.competitor else opponents
    for competitor in competitors:
        for algorithm in algorithms:
            for index in range(spec["num_experiments"]):
                execute_run(root / (algorithm + "_vs_" + competitor) / "run_{:02d}".format(index),
                            run_config(spec, algorithm, competitor, index, args.smoke_test))


if __name__ == "__main__":
    main()
