from collections import namedtuple

import gym
import numpy as np
from gym import spaces

from customer_model import CustomerSimulator
from simtools import get_history, price_grid, store_history, update_ref
from strategies import get_strat


HalfPeriod = namedtuple("HalfPeriod", "sales_a sales_b no_purchase revenue_a revenue_b")


class SpaceHandler(object):
    def __init__(self, min_p, max_p, step_size, undercut=100):
        self.min_p = min_p
        self.max_p = max_p
        self.step_size = step_size
        self.undercut = undercut

    def clip_price(self, price):
        return float(np.clip(price, self.min_p, self.max_p))


class ContinuousSpaceHandler(SpaceHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_length = (self.max_p - self.min_p) / 2

    def get_action_space(self):
        return spaces.Box(low=-self.space_length, high=self.space_length,
                          shape=(1,), dtype=np.float32)

    def fix_action(self, price, comp_price):
        return self.clip_price(round(self.min_p + self.space_length + price, 2))


class DiscreteSpaceHandler(SpaceHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prices = price_grid(self.min_p, self.max_p, self.step_size)

    def get_action_space(self):
        return spaces.Discrete(len(self.prices))

    def fix_action(self, price, comp_price):
        index = min(max(int(price), 0), len(self.prices) - 1)
        return float(self.prices[index])


class DiscreteRelativeSpaceHandler(DiscreteSpaceHandler):
    def get_action_space(self):
        return spaces.Discrete(len(self.prices) + 1)

    def fix_action(self, price, comp_price):
        if price == len(self.prices):
            return self.clip_price(comp_price - self.undercut)
        return super().fix_action(price, comp_price)


def get_action_space_handler(mode):
    return {
        "continuous": ContinuousSpaceHandler,
        "discrete_relative": DiscreteRelativeSpaceHandler,
    }.get(mode, DiscreteSpaceHandler)


class DuopolyMarket(object):
    def __init__(self, min_p, max_p, h, w, sim_params, step_size,
                 initial_price_a, initial_price_b, initial_reference_price, keep_history=True):
        self.min_p = min_p
        self.max_p = max_p
        self.step_size = step_size
        self.observation_space = spaces.Box(low=min_p, high=max_p,
                                            shape=(2,), dtype=np.float32)
        self.h = h
        self.w = w
        self.sim = CustomerSimulator(**sim_params)
        self.keep_history = keep_history
        self.initial_prices = tuple(float(np.clip(price, min_p, max_p))
                                    for price in (initial_price_a, initial_price_b))
        self.initial_reference_price = float(np.clip(initial_reference_price, min_p, max_p))
        self.reset_prices()
        self.clear_history()

    def reset_prices(self):
        self.price_a, self.price_b = self.initial_prices
        self.ref_p = self.initial_reference_price
        self.episodes = 0

    def clear_history(self):
        self.overall_profits_a = 0
        self.overall_profits_b = 0
        self.hist_a = get_history()
        self.hist_b = get_history()
        self.no_purchase = []
        self.customer_arrivals = []

    def observe(self, a=True):
        return np.asarray([self.ref_p, self.price_b if a else self.price_a], dtype=np.float32)

    def sell(self, first):
        duration = self.h if first else 1 - self.h
        sales_a, sales_b, no_purchase = self.sim(duration, self.price_a, self.price_b)
        if self.keep_history:
            (self.hist_a.sales_first if first else self.hist_a.sales_second).append(sales_a)
            (self.hist_b.sales_first if first else self.hist_b.sales_second).append(sales_b)
            self.no_purchase.append(no_purchase)
            self.customer_arrivals.append(sales_a + sales_b + no_purchase)
        return HalfPeriod(sales_a, sales_b, no_purchase,
                          sales_a * self.price_a, sales_b * self.price_b)

    def record_step(self, first, second):
        revenue_a = first.revenue_a + second.revenue_a
        revenue_b = first.revenue_b + second.revenue_b
        self.overall_profits_a += revenue_a
        self.overall_profits_b += revenue_b
        if self.keep_history:
            self.hist_a.profits.append(revenue_a)
            self.hist_b.profits.append(revenue_b)
            self.hist_a.sales.append(first.sales_a + second.sales_a)
            self.hist_b.sales.append(first.sales_b + second.sales_b)
        self.episodes += 1
        return revenue_a, revenue_b

    def store_history(self, file, losses=None):
        store_history(file, self.overall_profits_a, self.overall_profits_b,
                      self.hist_a, self.hist_b, losses=losses,
                      no_purchase=self.no_purchase, customer_arrivals=self.customer_arrivals)

    def display(self, text):
        print(text.format(self.overall_profits_a, self.overall_profits_b))


class MarketEnv(DuopolyMarket, gym.Env):
    """Two sellers with one learning agent and one scripted competitor."""
    metadata = {'render.modes': ['human']}

    def __init__(self, min_p, max_p, competitor, comp_params, sim_params, h, w,
                 mode="continuous", *args, step_size=100, initial_price_a=4000,
                 initial_price_b=4000, initial_reference_price=4000, undercut=100,
                 keep_history=True, **kwargs):
        super().__init__(min_p, max_p, h, w, sim_params, step_size,
                         initial_price_a, initial_price_b, initial_reference_price, keep_history)
        self.comp = get_strat(competitor)(**comp_params)
        self.mode = mode
        self.space_handler = get_action_space_handler(mode)(min_p, max_p, step_size, undercut)
        self.action_space = self.space_handler.get_action_space()

    def fix_action(self, price, comp_price):
        return self.space_handler.fix_action(price, comp_price)

    def step(self, action):
        self.price_a = self.fix_action(np.asarray(action).item(), self.price_b)
        if self.keep_history:
            self.hist_a.prices.append(self.price_a)
        self.ref_p = update_ref(self.ref_p, self.price_a, self.w)
        first = self.sell(first=True)

        self.price_b = self.space_handler.clip_price(self.comp(self.ref_p, self.price_a))
        if self.keep_history:
            self.hist_b.prices.append(self.price_b)
        self.ref_p = update_ref(self.ref_p, self.price_b, self.w)
        second = self.sell(first=False)

        revenue_a, revenue_b = self.record_step(first, second)
        sales_a = first.sales_a + second.sales_a
        sales_b = first.sales_b + second.sales_b
        no_purchase = first.no_purchase + second.no_purchase
        info = {
            "step": self.episodes,
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
        self.reset_prices()
        self.clear_history()
        return self.observe()

    def render(self, mode='human', close=False):
        pass


class DoubleMarketEnv(DuopolyMarket):
    """Shared duopoly with a Gym view for each learning agent."""

    def __init__(self, min_p, max_p, h, w, sim_params, mode="continuous",
                 mode_b="continuous", *args, step_size=100, initial_price_a=4000,
                 initial_price_b=4000, initial_reference_price=4000, undercut=100, **kwargs):
        super().__init__(min_p, max_p, h, w, sim_params, step_size,
                         initial_price_a, initial_price_b, initial_reference_price)
        self.space_handler_a = get_action_space_handler(mode)(min_p, max_p, step_size, undercut)
        self.space_handler_b = get_action_space_handler(mode_b)(min_p, max_p, step_size, undercut)
        self.a = None
        self.b = None

    def get_env(self, a=True):
        wrapping_env = self
        space_handler = self.space_handler_a if a else self.space_handler_b

        class Env(gym.Env):
            def __init__(self):
                self.action_space = space_handler.get_action_space()
                self.observation_space = wrapping_env.observation_space
                self.space_handler = space_handler

            def step(self, action):
                return wrapping_env.step(a, action)

            def reset(self):
                return wrapping_env.reset(a)

        return Env()

    def register_comp(self, comp, a=True):
        if a:
            self.a = comp
        else:
            self.b = comp

    def fix_action(self, price, a, comp_price):
        handler = self.space_handler_a if a else self.space_handler_b
        return handler.fix_action(price, comp_price)

    def finish_training(self):
        self.clear_history()

    def _half_step(self, a, action, first):
        price = self.fix_action(action, a, self.price_b if a else self.price_a)
        self.ref_p = update_ref(self.ref_p, price, self.w)
        if a:
            self.price_a = price
            self.hist_a.prices.append(price)
        else:
            self.price_b = price
            self.hist_b.prices.append(price)
        return self.sell(first)

    def half_step(self, a, action, first):
        period = self._half_step(a, action, first)
        return period.revenue_a, period.revenue_b

    def step(self, a, action):
        action_a = action if a else self.a.predict(self.observe()[None, :])[0]
        first = self._half_step(True, np.asarray(action_a).item(), True)
        action_b = self.b.predict(self.observe(False)[None, :])[0] if a else action
        second = self._half_step(False, np.asarray(action_b).item(), False)
        profits_a, profits_b = self.record_step(first, second)
        return self.observe(a), profits_a if a else profits_b, False, {}

    def reset(self, a):
        # Each learner resets at the start of its turn. Keep the shared training history.
        self.reset_prices()
        return self.observe(a)
