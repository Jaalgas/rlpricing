import random

import keras
import numpy as np
from keras.layers import Dense, Input, Lambda, Add, BatchNormalization, Dropout
from keras.models import Model
import tensorflow as tf
from keras.optimizers import Adam

GAMMA = 0.99
EPS = 1.0
MIN_EPS = 0.3
DECAY = 0.95
USE_DOUBLE = True
USE_DUELING = True
BATCH_SIZE = 32
NORM_FACT = 1000

HIDDEN_STRUCTURE = [
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    },
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    },
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    }
]

DUELING_HIDDEN_LAYERS = [
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    }
]

regularizers = {
    "l1": keras.regularizers.l1(0.001),
    "l2": keras.regularizers.l2(0.001)
}


class KerasQNetwork(object):
    def __init__(self,
                 num_inputs, num_outputs, hidden_structure=None,
                 use_dueling=USE_DUELING, dueling_hidden_layers=None, optimizer_args=None,
                 use_batch_norm=False, use_dropout=False, reg_target=None, reg_method=None):
        hidden_structure = hidden_structure or HIDDEN_STRUCTURE
        dueling_hidden_layers = dueling_hidden_layers or DUELING_HIDDEN_LAYERS
        optimizer_args = optimizer_args or {}
        input_layer = Input(shape=(num_inputs, ))

        additional_layer_args = {}
        if reg_target:
            additional_layer_args["{}_regularizer".format(reg_target)] = regularizers.get(reg_method, None)

        append_batchnorm = BatchNormalization() if use_batch_norm else lambda l: l
        append_dropout = Dropout(0.2) if use_dropout else lambda l: l

        def build_dense(layer, args):
            layer = Dense(**args, **additional_layer_args)(layer)
            layer = append_batchnorm(layer)
            return append_dropout(layer)

        def build_output_layer(num_o, input, name=None):
            return Dense(
                num_o,
                name=name,
                activation="linear",
                use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_normal"
            )(input)

        layer = input_layer
        for el in hidden_structure:
            layer = build_dense(layer, el)

        if use_dueling:
            state_layer = layer
            for el in dueling_hidden_layers:
                state_layer = build_dense(state_layer, el)
            state_layer = build_output_layer(1, state_layer)

            action_layer = layer
            for el in dueling_hidden_layers:
                action_layer = build_dense(action_layer, el)
            action_layer = build_output_layer(num_outputs, action_layer)

            norm_action = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keep_dims=True))(action_layer)
            layer = Add(name="output")([state_layer, norm_action])

        if not use_dueling:
            layer = build_output_layer(num_outputs, layer, "output")

        model = Model(inputs=input_layer, outputs=layer)
        model.compile(optimizer=Adam(**optimizer_args), loss=["mse"])
        self.model = model

    def fit_to(self, inputs, target_q, weights):
        return self.model.train_on_batch(inputs, target_q, sample_weight=weights)

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions

    def copy_from(self, network):
        self.model.set_weights(network.model.get_weights())


class QLModel(object):

    def __init__(self, num_inputs, num_outputs, eps=EPS, decay=DECAY, min_eps=MIN_EPS,
                 network_params=None, use_double=USE_DOUBLE, gamma=GAMMA):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.use_double = use_double
        params = network_params or {}
        self.network = KerasQNetwork(num_inputs, num_outputs, **params)
        self.target = KerasQNetwork(num_inputs, num_outputs, **params) if use_double else None
        self.eps = eps
        self.gamma = gamma
        self.decay = decay
        self.min_eps = min_eps
        self.losses = []

    def predict(self, inputs, use_target=False):
        network = self.target if use_target else self.network
        return network.predict(inputs)

    def predict_values(self, inputs, use_target=False):
        if np.any(np.isnan(inputs)):
            print("NaN in input")

        predictions = self.predict(inputs, use_target)

        predictions = predictions[0]
        if np.any(np.isnan(predictions)):
            print("NaN in output")

        return np.argmax(predictions), predictions

    def copy_to_target(self):
        self.target.copy_from(self.network)

    def train(self, tuples, weights):
        indexes = list(range(len(tuples)))
        random.shuffle(indexes)
        print('Training on the last ' + str(
            len(tuples)) + ' samples.')

        if self.use_double:
            self.copy_to_target()

        losses = []
        batch_x = np.zeros((BATCH_SIZE, self.num_inputs))
        batch_y = np.zeros((BATCH_SIZE, self.num_outputs))
        batch_weights = np.zeros((BATCH_SIZE,))
        loss_updates = np.zeros(len(tuples))
        i = 0
        for idx in indexes:
            tup = tuples[idx]
            weight = float(np.asarray(weights[idx]).item())

            # Find predicted values
            _, predicted = self.predict_values(
                tup.s
            )

            # Find predictions for next step
            next_chose, _ = self.predict_values(
                tup.s2
            )
            _, next_predicted = self.predict_values(
                tup.s2,
                self.use_double
            )

            # Calculate at which output the price has actually been.
            pos = tup.a

            # Rewrite this value with the actual result value.
            target_value = self.normalize_reward(tup.r) + self.gamma * next_predicted[next_chose]
            loss_updates[idx] = abs(target_value - predicted[pos])
            predicted[pos] = target_value

            batch_x[i] = tup.s[0]
            batch_y[i] = predicted
            batch_weights[i] = weight

            i += 1

            if i == BATCH_SIZE:
                losses.append(self.network.fit_to(batch_x, batch_y, batch_weights))
                i = 0

        if i > 0:
            losses.append(self.network.fit_to(batch_x[:i], batch_y[:i], batch_weights[:i]))

        self.losses.append(sum(losses)/len(losses))

        self.eps *= self.decay
        if self.eps < self.min_eps:
            self.eps = self.min_eps

        return loss_updates

    def normalize_reward(self, r):
        return r / NORM_FACT

    def get_price(self, features):
        if np.random.uniform() < self.eps:
            return round(np.random.uniform(high=self.num_outputs - 1))
        return self.predict_values(features)[0]
