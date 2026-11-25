import numpy as np

from .store import UniformStore, PrioritizedStore

from gymql.ql import QLModel as QNetwork


class QLModel(object):
    def __init__(self, _, env, **kwargs):
        self.num_prices = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.min_price = float(env.observation_space.low[-1])
        self.price_scale = float(env.observation_space.high[-1]) - self.min_price
        relative_action_index = None
        if kwargs.get("use_relative"):
            handler = env.get_attr("space_handler")[0]
            relative_action_index = lambda state: self.relative_action_index(state, handler)
        self.q = QNetwork(self.num_features, self.num_prices,
                          relative_action_index=relative_action_index, **kwargs)
        self.periodic_profits = []
        self.last_train = 0
        self.overall_eps = 0
        self.train_eps = kwargs.get("training_eps", 128)
        self.sample_size = kwargs.get("sample_size", 256)
        self.env = env
        store_p = {"size": 16384, **kwargs.get("store_params", {})}
        self.store = UniformStore(**store_p)\
            if kwargs.get("store_type", "uniform") == "uniform"\
            else PrioritizedStore(**store_p)

    def predict(self, obs):
        states = self.normalize_state(np.atleast_2d(obs))
        return np.argmax(self.q.predict(states), axis=1), None

    def normalize_state(self, state):
        return ((state - self.min_price) / self.price_scale) - 0.5

    def relative_action_index(self, state, handler):
        opponent_price = float(state[0, 1] + 0.5) * self.price_scale + self.min_price
        price = handler.fix_action(self.num_prices - 1, opponent_price)
        return int(np.argmin(np.abs(handler.prices - price)))

    def save(self, path):
        self.q.network.model.save_weights(str(path))

    def load_weights(self, path):
        self.q.network.model.load_weights(str(path))

    def learn(self, total_timesteps, log_interval, callback):
        state = self.normalize_state(self.env.reset())

        for i in range(0, total_timesteps):
            price = self.learn_single(state)
            last_state = state
            state, rewards, done, info = self.env.step([price])
            state = self.normalize_state(state)
            self.store.store(last_state, price, rewards, state)
            if done:
                break
            if i % log_interval == 0:
                if callback({'step': i, 'losses': self.q.losses}, {}) is False:
                    break

    def learn_single(self, features):
        if self.last_train % self.train_eps == 0 and self.overall_eps >= self.store.size:
            self.last_train = 0
            samples, indices, weights = self.store.get_sample(self.sample_size)
            updates = self.q.train(samples, weights)
            self.store.update_weights(indices, updates)
        price = int(self.q.get_price(features))
        self.last_train += 1
        self.overall_eps += 1
        return price

    def get_periodic_profits(self):
        return self.periodic_profits
