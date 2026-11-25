import json
from collections import namedtuple

import numpy as np
from collections.abc import Mapping


class CustomerSimulator(object):
    def __init__(self, params, factor=80):
        self.params = np.asarray(params)
        self.num_comps = 1
        self.factor = factor

    def get_lambda(self, length, a, p, r):
        features = np.array([
            1,
            (1 + ((1 if p < a else 0) + (1 if p <= a else 0)) / 2),
            a - p,
            self.num_comps,
            (a + p) / 2,
            a - r
        ])
        return (length * np.exp(np.dot(self.params, features)) / (
                1 + np.exp(np.dot(self.params, features)))) * self.factor

    def __call__(self, length, own_price, comp_price, ref_price):
        return np.random.poisson(self.get_lambda(length, own_price, comp_price, ref_price))


DEFAULT_PARAMS = [-3.89, -0.56, -0.01, 0.07, -0.03, 0]
DEFAULT_PARAMS_REF = [-3.89, -0.56, -0.01, 0.07, -0.03, -0.01]
DEFAULT_SIM = CustomerSimulator(DEFAULT_PARAMS)

history_fields = ['prices', 'sales', 'sales_first', 'sales_second', 'profits', 'metadata']
History = namedtuple('History', history_fields)


def get_history():
    return History([], [], [], [], [], [])


def update_ref(old_ref, price, w=0.9):
    return (old_ref * w) + ((1 - w) * price)


def price_grid(min_p, max_p, step_size):
    if step_size <= 0 or max_p <= min_p:
        raise ValueError("Expected positive price step and increasing market bounds")
    prices = min_p + np.arange(int(np.ceil((max_p - min_p) / step_size))) * step_size
    return np.append(prices[prices < max_p], max_p)


def store_history(path, a_profits, b_profits, history_a, history_b, losses=None,
                  no_purchase=None, customer_arrivals=None):
    data = {
        "a_profits": a_profits,
        "b_profits": b_profits,
        "a": history_a._asdict(),
        "b": history_b._asdict(),
        "losses": losses
    }
    if customer_arrivals is not None:
        data.update(no_purchase=no_purchase, customer_arrivals=customer_arrivals)
    with open(path, "w") as file:
        json.dump(data, file)


def store_oligopoly_history(path, histories):
    data = [
        a._asdict() for a in histories
    ]
    with open(path, "w") as file:
        json.dump(data, file)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
