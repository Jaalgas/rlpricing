import numpy as np

from .store import PrioritizedStore

from gymql.ql import QLModel as QNetwork


class QLModel(object):
    def __init__(self, env, *, store_type="prioritized", store_params=None,
                 training_eps=128, sample_size=256, **model_params):
        if store_type != "prioritized":
            raise ValueError("Q-learning uses prioritized replay")
        self.num_prices = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.min_price = float(env.observation_space.low[-1])
        self.price_scale = float(env.observation_space.high[-1]) - self.min_price
        self.q = QNetwork(self.num_features, self.num_prices, **model_params)
        self.last_train = 0
        self.overall_eps = 0
        self.train_eps = training_eps
        self.sample_size = sample_size
        self.env = env
        self.store = PrioritizedStore(**{"size": 16384, **(store_params or {})})

    def predict(self, obs):
        states = self.normalize_state(np.atleast_2d(obs))
        return np.argmax(self.q.predict(states), axis=1), None

    def normalize_state(self, state):
        return ((state - self.min_price) / self.price_scale) - 0.5

    def save(self, path):
        self.q.network.model.save_weights(str(path))

    def load_weights(self, path):
        self.q.network.model.load_weights(str(path))

    def learn(self, total_timesteps):
        state = self.normalize_state(self.env.reset())

        for _ in range(total_timesteps):
            price = self.learn_single(state)
            last_state = state
            state, rewards, done, _ = self.env.step([price])
            state = self.normalize_state(state)
            self.store.store(last_state, price, rewards, state)
            if done:
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
