from collections.abc import Mapping

import numpy as np


def update_ref(old_ref, price, w=0.9):
    return (old_ref * w) + ((1 - w) * price)


def price_grid(min_p, max_p, step_size):
    if step_size <= 0 or max_p <= min_p:
        raise ValueError("Expected positive price step and increasing market bounds")
    prices = min_p + np.arange(int(np.ceil((max_p - min_p) / step_size))) * step_size
    return np.append(prices[prices < max_p], max_p)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
