"""Market and replay regression tests for the eSIM experiments."""
import json
from pathlib import Path
import unittest

import numpy as np

from customer_model import CustomerSimulator
from environments.duopoly import MarketEnv
from gymql.store import PrioritizedStore, SumTree
from run_analysis import describe, evaluation_metrics, summarize
from strategies import CartelTwoBound, TwoBound


CONFIG = json.loads(Path(__file__).with_name("gym_esim.json").read_text(encoding="utf-8"))


class CustomerTests(unittest.TestCase):
    def simulator(self, seed=42):
        return CustomerSimulator(**CONFIG["gym_params"]["sim_params"], seed=seed)

    def test_generation_and_income_brackets(self):
        sim = self.simulator()
        income, theta_p, theta_d = sim.generate_customers(1000)
        self.assertLess(abs(len(income) - 100000), 2000)
        proportions = np.histogram(income, bins=sim.bounds)[0] / len(income)
        np.testing.assert_allclose(proportions, sim.shares, atol=0.005)
        self.assertTrue(np.all(theta_p > 0) and np.all(theta_d > 0))
        self.assertEqual(len(sim.generate_customers(0)[0]), 0)

    def test_probability_and_customer_accounting(self):
        sim = self.simulator()
        counts = sim(2, 4000, 4000)
        choices = sim.last_batch["choice"]
        probabilities = sim.last_batch["probabilities"]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[:, 0], probabilities[:, 1])
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(len(sim.last_batch["income"]), sum(counts))
        np.testing.assert_array_equal(np.bincount(choices, minlength=3), counts)

    def test_continuous_arrivals_and_reproducibility(self):
        a, b = self.simulator(), self.simulator()
        first = [a(0.5, 3500, 4000) for _ in range(30)]
        second = [b(0.5, 9000, 4000) for _ in range(30)]
        arrivals = list(map(sum, first))
        self.assertEqual(arrivals, list(map(sum, second)))
        self.assertGreater(min(arrivals), 0)
        self.assertGreater(len(set(arrivals)), 1)
        c = self.simulator()
        self.assertEqual(first, [c(0.5, 3500, 4000) for _ in range(30)])


class MarketTests(unittest.TestCase):
    def market(self, mode="continuous", **options):
        params = dict(CONFIG["gym_params"], mode=mode, competitor="fixed",
                      reward_mode="own_revenue",
                      comp_params=CONFIG["competitors"]["fixed"])
        params.update(options)
        return MarketEnv(**params)

    def test_shared_customers_and_revenue(self):
        env = self.market(h=0.25)
        batch_calls = []

        def shared_batch(duration, a, b):
            batch_calls.append((duration, a, b))
            return (3, 5, 7) if duration == 0.25 else (11, 13, 17)

        env.sim = shared_batch
        obs, reward, done, row = env.step([0])
        self.assertEqual(batch_calls, [(0.25, 5500, 4000), (0.75, 5500, 4000)])
        self.assertEqual((row["sales_a"], row["sales_b"], row["no_purchase"]), (14, 18, 24))
        self.assertEqual((row["first_half"]["sales_b"], row["second_half"]["sales_b"]), (5, 13))
        self.assertEqual((row["total_customers"], row["revenue_a"], row["revenue_b"]), (56, 77000, 72000))
        self.assertEqual(reward, row["reward"])
        self.assertEqual(reward, env.total_revenue_a)
        self.assertEqual(row["revenue_b"], env.total_revenue_b)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(done)

    def test_mixed_reward_weights_revenue_and_competitive_lead(self):
        params = CONFIG["competitors"]["twobound"]
        mixed = self.market(competitor="twobound", comp_params=params,
                            reward_mode="revenue_mix", revenue_weight=0.7,
                            competitive_weight=0.3)
        baseline = self.market(competitor="twobound", comp_params=params)
        mixed.reset(seed=73)
        baseline.reset(seed=73)
        for action in ([0], [-4500], [4500]):
            observation, reward, _, row = mixed.step(action)
            base_observation, base_reward, _, base_row = baseline.step(action)
            np.testing.assert_array_equal(observation, base_observation)
            self.assertEqual(base_reward, base_row["revenue_a"])
            self.assertEqual(row["reward"], reward)
            self.assertAlmostEqual(
                reward, 0.7 * row["revenue_a"] + 0.3 * (row["revenue_a"] - row["revenue_b"]))
            self.assertEqual((row["revenue_a"], row["revenue_b"]),
                             (base_row["revenue_a"], base_row["revenue_b"]))

    def test_mixed_reward_can_prioritize_beating_the_competitor(self):
        params = CONFIG["competitors"]["twobound"]
        env = self.market(competitor="twobound", comp_params=params,
                          reward_mode="revenue_mix", revenue_weight=0.3,
                          competitive_weight=0.7)
        env.reset(seed=73)
        _, reward, _, row = env.step([0])
        self.assertAlmostEqual(
            reward, 0.3 * row["revenue_a"] + 0.7 * (row["revenue_a"] - row["revenue_b"]))

    def test_revenue_lead_uses_both_half_periods(self):
        params = CONFIG["competitors"]["twobound"]
        env = self.market(competitor="twobound", comp_params=params, reward_mode="revenue_lead")
        baseline = self.market(competitor="twobound", comp_params=params)
        env.reset(seed=42)
        baseline.reset(seed=42)
        for action in ([0], [-4500], [4500]):
            obs, reward, _, row = env.step(action)
            base_obs, base_reward, _, base_row = baseline.step(action)
            np.testing.assert_array_equal(obs, base_obs)
            self.assertEqual(row["reward"], reward)
            self.assertEqual(reward, row["revenue_a"] - row["revenue_b"])
            self.assertEqual(dict(row, reward=base_reward), base_row)

    def test_reactive_prices_and_revenue_by_half(self):
        for competitor, action, price_a, response in [
            ("twobound", 0, 5500, 5400), ("twobound", -4500, 1000, 10000),
            ("cartel", 1100, 6600, 6600), ("cartel", 0, 5500, 5400),
        ]:
            with self.subTest(competitor=competitor, action=action):
                env = self.market(h=0.25, competitor=competitor,
                                  comp_params=CONFIG["competitors"][competitor])
                calls = []

                def shared_batch(duration, a, b):
                    calls.append((duration, a, b))
                    return (3, 5, 7) if duration == 0.25 else (11, 13, 17)

                env.sim = shared_batch
                obs, reward, _, row = env.step([action])
                self.assertEqual(calls, [(0.25, price_a, 4000), (0.75, price_a, response)])
                self.assertEqual(row["first_half"]["price_b"], 4000)
                self.assertEqual(row["second_half"]["price_b"], response)
                self.assertEqual(row["revenue_b"], 5 * 4000 + 13 * response)
                self.assertEqual(reward, 14 * price_a)
                self.assertEqual(row["total_customers"], 56)
                self.assertAlmostEqual(row["reference_price"],
                                       0.9 * (0.9 * 4000 + 0.1 * price_a) + 0.1 * response)
                self.assertEqual(obs[1], response)
                env.step([action])
                self.assertEqual(calls[2][2], response)
                np.testing.assert_array_equal(env.reset(), [4000, 4000])

    def test_historical_strategy_rules(self):
        two_bound = TwoBound(2000, 6000, 100)
        self.assertEqual(two_bound(4000, 2100), 2000)
        self.assertEqual(two_bound(4000, 2099), 6000)
        self.assertEqual(two_bound(4000, 10000), 9900)
        self.assertEqual(TwoBound(2000, 6000, 100, True)(4000, 10000), 6000)
        cartel = CartelTwoBound(1000, 10000, 100, 6600)
        self.assertEqual(cartel(4000, 6600.4), 6600.4)
        self.assertEqual(cartel(4000, 6600.6), 6500.6)

    def test_action_shapes_and_bounds(self):
        continuous = self.market()
        self.assertEqual(continuous.action_space.shape, (1,))
        self.assertEqual(continuous.observation_space.shape, (2,))
        self.assertEqual(continuous.reset().dtype, np.float32)
        for action, expected in [(-9000, 1000), (-4500, 1000), (4500, 10000), (9000, 10000)]:
            self.assertEqual(continuous.step([action])[3]["price_a"], expected)
        discrete = self.market("discrete")
        self.assertEqual(discrete.action_space.n, 91)
        self.assertEqual([discrete.decode_action(i) for i in range(91)], list(range(1000, 10001, 100)))
        self.assertEqual(discrete.step(np.int64(90))[3]["price_a"], 10000)

    def test_reset_clears_training_history_and_reseeds_evaluation(self):
        env = self.market()
        env.reset(seed=15)
        training_row = env.step([10])[3]
        np.testing.assert_array_equal(env.reset(seed=16), [4000, 4000])
        self.assertEqual(env.price_a, 4000)
        self.assertEqual(env.total_revenue_a, 0)
        self.assertEqual(env.total_revenue_b, 0)
        self.assertEqual(env.steps, 0)
        evaluation_row = env.step([10])[3]
        self.assertEqual(evaluation_row["step"], 1)
        self.assertEqual(evaluation_row["price_b"], 4000)
        self.assertNotEqual(training_row, evaluation_row)
        env.reset(seed=16)
        self.assertEqual(evaluation_row, env.step([10])[3])

    def test_step_records_are_independent(self):
        env = self.market()
        _, reward, _, first = env.step([0])
        _, _, _, second = env.step([0])
        self.assertEqual(first["reward"], reward)
        self.assertEqual((first["step"], second["step"]), (1, 2))
        self.assertEqual(env.total_revenue_a, first["revenue_a"] + second["revenue_a"])


class ReplayAndAnalysisTests(unittest.TestCase):
    def test_prioritized_sampling(self):
        tree = SumTree(2)
        tree.add_new_priority(1, "left")
        tree.add_new_priority(1, "right")
        self.assertEqual(tree.get_leaf(0.5)[2], "left")
        self.assertEqual(tree.get_leaf(1.5)[2], "right")
        store = PrioritizedStore(1)
        store.store("state", 90, np.array([5]), "next")
        samples, indices, weights = store.get_sample(1)
        self.assertEqual(samples[0].a, 90)
        self.assertTrue(np.isfinite(weights).all())
        store.update_weights(indices, [0.02])
        self.assertGreater(store.tree.root_priority, 0)

    def test_matched_summaries_only_compare_the_same_competitor(self):
        results = []
        for strategy, revenue_a in (("fixed", 100), ("twobound", 200), ("cartel", 400)):
            for algorithm, gain in (("ql", 0), ("sac", 20)):
                config = dict(algorithm=algorithm, run=0, evaluation_seed=42, eval_timesteps=1,
                              package_gb=10, gym_params=dict(competitor=strategy, mode=algorithm))
                row = dict(revenue_a=revenue_a + gain, revenue_b=100, sales_a=1, sales_b=1,
                           no_purchase=0, total_customers=2)
                results.append((None, config, evaluation_metrics([row], config)))
        paired = {row["comparison"]: row["mean"] for row in summarize(results)
                  if row["comparison"].startswith("matched") and row["metric"] == "revenue_a"}
        self.assertEqual(paired, {"matched_sac_minus_ql_vs_" + strategy: 20
                                  for strategy in ("fixed", "twobound", "cartel")})

    def test_undefined_metrics_arent_filled(self):
        config = dict(algorithm="ql", run=0, evaluation_seed=42, eval_timesteps=1,
                      gym_params={"competitor": "fixed"})
        result = evaluation_metrics([], config)
        for field in ("revenue_advantage_pct", "market_share_a_pct", "market_share_b_pct",
                      "conversion_rate_pct", "average_price_a", "average_price_b"):
            self.assertIsNone(result[field])
        self.assertIsNone(describe([42])["ci95_low"])
        self.assertLess(describe([35, 42])["ci95_low"], 35)


if __name__ == "__main__":
    unittest.main()
