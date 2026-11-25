"""Market and replay regression tests for the eSIM experiments."""
import json
from pathlib import Path
import unittest

import numpy as np

from customer_model import CustomerSimulator
from environments.duopoly import MarketEnv
from gymql.store import PrioritizedStore, SumTree
from run_analysis import describe, evaluation_metrics


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
        return MarketEnv(**dict(CONFIG["gym_params"], mode=mode, **options))

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
        self.assertEqual(reward, env.overall_profits_a)
        self.assertEqual(row["revenue_b"], env.overall_profits_b)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(done)

    def test_action_shapes_and_bounds(self):
        continuous = self.market()
        self.assertEqual(continuous.action_space.shape, (1,))
        self.assertEqual(continuous.observation_space.shape, (2,))
        self.assertEqual(continuous.reset().dtype, np.float32)
        for action, expected in [(-9000, 1000), (-4500, 1000), (4500, 10000), (9000, 10000)]:
            self.assertEqual(continuous.step([action])[3]["price_a"], expected)
        discrete = self.market("discrete")
        self.assertEqual(discrete.action_space.n, 91)
        self.assertEqual([discrete.fix_action(i, 4000) for i in range(91)], list(range(1000, 10001, 100)))
        self.assertEqual(discrete.step(np.int64(90))[3]["price_a"], 10000)

    def test_reset_clears_training_history_and_reseeds_evaluation(self):
        env = self.market()
        env.reset(seed=15)
        training_row = env.step([10])[3]
        np.testing.assert_array_equal(env.reset(seed=16), [4000, 4000])
        self.assertEqual(env.price_a, 4000)
        self.assertEqual(env.overall_profits_a, 0)
        self.assertEqual(env.overall_profits_b, 0)
        self.assertFalse(env.hist_a.sales or env.hist_b.sales or env.customer_arrivals)
        evaluation_row = env.step([10])[3]
        self.assertEqual(evaluation_row["step"], 1)
        self.assertEqual(evaluation_row["price_b"], 4000)
        self.assertNotEqual(training_row, evaluation_row)
        env.reset(seed=16)
        self.assertEqual(evaluation_row, env.step([10])[3])

    def test_streaming_mode_keeps_no_step_arrays(self):
        env = self.market(keep_history=False)
        _, reward, _, row = env.step([0])
        self.assertEqual(reward, row["revenue_a"])
        self.assertFalse(env.hist_a.prices or env.hist_a.sales or env.customer_arrivals)


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

    def test_undefined_metrics_arent_filled(self):
        config = dict(algorithm="ql", run=0, evaluation_seed=42, eval_timesteps=1)
        result = evaluation_metrics([], config)
        for field in ("revenue_advantage_pct", "market_share_a_pct", "market_share_b_pct",
                      "conversion_rate_pct", "average_price_a", "average_price_b"):
            self.assertIsNone(result[field])
        self.assertIsNone(describe([42])["ci95_low"])
        self.assertLess(describe([35, 42])["ci95_low"], 35)


if __name__ == "__main__":
    unittest.main()
