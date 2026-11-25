from collections import namedtuple

import gym
import numpy as np
from gym import spaces

from customer_model import CustomerSimulator
from simtools import price_grid, update_ref
from strategies import Fixed


HalfPeriod = namedtuple("HalfPeriod", "sales_a sales_b no_purchase revenue_a revenue_b")


class MarketEnv(gym.Env):
    """Two sellers with one learning agent and one fixed-price competitor."""

    def __init__(self, min_p, max_p, competitor, comp_params, sim_params, h, w,
                 mode="continuous", *, step_size=100, initial_price_a=4000,
                 initial_price_b=4000, initial_reference_price=4000):
        self.min_p = min_p
        self.max_p = max_p
        self.mode = mode
        if mode == "discrete":
            self.prices = price_grid(min_p, max_p, step_size)
            self.action_space = spaces.Discrete(len(self.prices))
        elif mode == "continuous":
            self.half_range = (max_p - min_p) / 2
            self.action_space = spaces.Box(low=-self.half_range, high=self.half_range,
                                          shape=(1,), dtype=np.float32)
        else:
            raise ValueError("Expected discrete or continuous pricing actions")
        self.observation_space = spaces.Box(low=min_p, high=max_p,
                                           shape=(2,), dtype=np.float32)

        if competitor != "fixed":
            raise ValueError("The competitor must use fixed pricing")
        self.competitor = Fixed(**comp_params)
        if self.competitor.price != initial_price_b or not min_p <= initial_price_b <= max_p:
            raise ValueError("The fixed competitor price must equal its opening price and be within bounds")
        self.initial_price_a = float(np.clip(initial_price_a, min_p, max_p))
        self.initial_reference_price = float(np.clip(initial_reference_price, min_p, max_p))
        self.h = h
        self.w = w
        self.sim = CustomerSimulator(**sim_params)
        self.reset()

    def decode_action(self, action):
        if self.mode == "discrete":
            index = min(max(int(action), 0), len(self.prices) - 1)
            return float(self.prices[index])
        price = round(self.min_p + self.half_range + action, 2)
        return float(np.clip(price, self.min_p, self.max_p))

    def observe(self):
        return np.asarray([self.ref_p, self.price_b], dtype=np.float32)

    def sell(self, duration):
        sales_a, sales_b, no_purchase = self.sim(duration, self.price_a, self.price_b)
        return HalfPeriod(sales_a, sales_b, no_purchase,
                          sales_a * self.price_a, sales_b * self.price_b)

    def step(self, action):
        self.price_a = self.decode_action(np.asarray(action).item())
        self.ref_p = update_ref(self.ref_p, self.price_a, self.w)
        first = self.sell(self.h)

        self.ref_p = update_ref(self.ref_p, self.price_b, self.w)
        second = self.sell(1 - self.h)

        revenue_a = first.revenue_a + second.revenue_a
        revenue_b = first.revenue_b + second.revenue_b
        self.total_revenue_a += revenue_a
        self.total_revenue_b += revenue_b
        self.steps += 1
        sales_a = first.sales_a + second.sales_a
        sales_b = first.sales_b + second.sales_b
        no_purchase = first.no_purchase + second.no_purchase
        info = {
            "step": self.steps,
            "price_a": self.price_a,
            "price_b": self.price_b,
            "sales_a": sales_a,
            "sales_b": sales_b,
            "no_purchase": no_purchase,
            "total_customers": sales_a + sales_b + no_purchase,
            "revenue_a": revenue_a,
            "revenue_b": revenue_b,
            "reward": revenue_a,
            "reference_price": self.ref_p,
            "first_half": first._asdict(),
            "second_half": second._asdict(),
        }
        return self.observe(), revenue_a, False, info

    def reset(self, seed=None):
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.price_a = self.initial_price_a
        self.price_b = self.competitor.price
        self.ref_p = self.initial_reference_price
        self.steps = 0
        self.total_revenue_a = 0
        self.total_revenue_b = 0
        return self.observe()
