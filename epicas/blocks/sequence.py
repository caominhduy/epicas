"""
LSTM, Bidirectional LSTM (Bi-LSTM), and GRU models

Input: Multivariate, multi-step, multiple time-series
-> Output: Multivariate, single-step, multiple time-series
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, optimizers
from tensorflow import keras
from tqdm import tqdm
import os
from datetime import date, datetime, timedelta

from .input import InputLayer

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # comment out to use GPU(s)


class AutoLSTM(InputLayer):
    def __init__(self, EpiData, model=None, **kwargs):
        super().__init__(EpiData)
        self.model = model

        self.n_timesteps = 14

    def fit(self, verbose=True):
        """Fitting the model and automatically search for best hyperparameters
        within expected range

        # Arguments:
            - verbose: bool. Whether function should print relavant details
            about model

        """
        X_train = np.array(
            np.hsplit(self.X_train.to_numpy(), len(self.locations)))
        Y_train = X_train

        X_val = np.array(np.hsplit(self.X_val.to_numpy(), len(self.locations)))
        Y_val = X_val

        X_test = np.array(
            np.hsplit(self.X_test.to_numpy(), len(self.locations)))
        Y_test = X_test

        n_timesteps = self.n_timesteps  # number of observation to look back

        self.model = self._build_LSTM_model(X_train, n_timesteps)

        self.model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss='mae',
            metrics=['accuracy'])

        if verbose:
            self.model.summary()

        for i in tqdm(range(n_timesteps, X_train.shape[1]), desc='LSTM Fitting'):
            X, Y = X_train[:, (i - n_timesteps):i, :], Y_train[:, i:i + 1, :]

            history = self.model.fit(X, Y, shuffle=False, verbose=0)

        for i in range(n_timesteps, X_val.shape[1]):
            X, Y = X_val[:, (i - n_timesteps):i, :], Y_val[:, i:i + 1, :]

            self.model.evaluate(X, Y, verbose=0)

        return AutoLSTM(self.EpiData, self.model)

    def predict(self, target_date):
        """Predict data from fitted model

        # Argument:
            - target_date: str or datetime
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        predict_dates = []

        if target_date <= self.dates_test[-1]:

            X = np.array(np.hsplit(self.X_train.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_train[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_train[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        elif target_date > self.dates_test[-1]:
            X = np.array(np.hsplit(self.X_test.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_test[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_test[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        predict_values = list(X[:, -len(predict_dates):, 0])

        preds = pd.DataFrame(np.array(predict_values).T,
                    columns = self.Y_val.columns.values.tolist())

        preds['date'] = predict_dates

        preds = preds.melt(id_vars=['date'], \
                            var_name='location', \
                            value_name=self.target)\
                    .reindex(['location', 'date', self.target], axis=1)

        preds.date = preds.date.astype('datetime64[D]')

        return preds

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        return AutoLSTM(self.EpiData, self.model)

    @staticmethod
    def _build_LSTM_model(X, n_timesteps):
        # hyperparameters
        n_features = X.shape[-1]
        hidden_units = 64

        # functional API
        inputs = Input(shape=(n_timesteps, n_features))
        lstm = layers.LSTM(hidden_units, return_sequences=True)(inputs)
        lstm = layers.LSTM(hidden_units)(lstm)
        model = layers.Dense(n_features)(lstm)

        return Model(inputs, model)


class AutoGRU(InputLayer):
    def __init__(self, EpiData, model=None, **kwargs):
        super().__init__(EpiData)
        self.model = model
        self.n_timesteps = 14

    def fit(self, verbose=True):
        """Fitting the model and automatically search for best hyperparameters
        within expected range

        # Arguments:
            - verbose: bool. Whether function should print relavant details
            about model

        """
        X_train = np.array(
            np.hsplit(self.X_train.to_numpy(), len(self.locations)))
        Y_train = X_train

        X_val = np.array(np.hsplit(self.X_val.to_numpy(), len(self.locations)))
        Y_val = X_val

        X_test = np.array(
            np.hsplit(self.X_test.to_numpy(), len(self.locations)))
        Y_test = X_test

        n_timesteps = self.n_timesteps  # number of observation to look back

        self.model = self._build_GRU_model(X_train, n_timesteps)

        self.model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss='mae',
            metrics=['accuracy'])

        if verbose:
            self.model.summary()

        for i in tqdm(range(n_timesteps, X_train.shape[1]), desc='GRU Fitting'):
            X, Y = X_train[:, (i - n_timesteps):i, :], Y_train[:, i:i + 1, :]

            history = self.model.fit(X, Y, shuffle=False, verbose=0)

        for i in range(n_timesteps, X_val.shape[1]):
            X, Y = X_val[:, (i - n_timesteps):i, :], Y_val[:, i:i + 1, :]

            self.model.evaluate(X, Y, verbose=0)

        return AutoGRU(self.EpiData, self.model)

    def predict(self, target_date):
        """Predict data from fitted model

        # Argument:
            - target_date: str or datetime
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        predict_dates = []

        if target_date <= self.dates_test[-1]:

            X = np.array(np.hsplit(self.X_train.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_train[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_train[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        else:
            X = np.array(np.hsplit(self.X_test.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_test[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_test[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        predict_values = list(X[:, -len(predict_dates):, 0])

        preds = pd.DataFrame(np.array(predict_values).T,
                    columns = self.Y_val.columns.values.tolist())

        preds['date'] = predict_dates

        preds = preds.melt(id_vars=['date'], \
                            var_name='location', \
                            value_name=self.target)\
                    .reindex(['location', 'date', self.target], axis=1)

        preds.date = preds.date.astype('datetime64[D]')

        return preds

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        return AutoGRU(self.EpiData, self.model)

    @staticmethod
    def _build_GRU_model(X, n_timesteps):
        # hyperparameters
        n_features = X.shape[-1]
        hidden_units = 64

        # functional API
        inputs = Input(shape=(n_timesteps, n_features))
        gru = layers.GRU(hidden_units, return_sequences=True)(inputs)
        gru = layers.GRU(hidden_units)(gru)
        model = layers.Dense(n_features)(gru)

        return Model(inputs, model)


class AutoBiLSTM(InputLayer):
    def __init__(self, EpiData, model=None, **kwargs):
        super().__init__(EpiData)
        self.model = model
        self.n_timesteps = 14

    def fit(self, verbose=True):
        """Fitting the model and automatically search for best hyperparameters
        within expected range

        # Arguments:
            - verbose: bool. Whether function should print relavant details
            about model

        """
        X_train = np.array(
            np.hsplit(self.X_train.to_numpy(), len(self.locations)))
        Y_train = X_train

        X_val = np.array(np.hsplit(self.X_val.to_numpy(), len(self.locations)))
        Y_val = X_val

        X_test = np.array(
            np.hsplit(self.X_test.to_numpy(), len(self.locations)))
        Y_test = X_test

        n_timesteps = self.n_timesteps  # number of observation to look back

        self.model = self._build_BiLSTM_model(X_train, n_timesteps)

        self.model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss='mae',
            metrics=['accuracy'])

        if verbose:
            self.model.summary()

        for i in tqdm(range(n_timesteps, X_train.shape[1]), desc='Bidirectional LSTM Fitting'):
            X, Y = X_train[:, (i - n_timesteps):i, :], Y_train[:, i:i + 1, :]

            history = self.model.fit(X, Y, shuffle=False, verbose=0)

        for i in range(n_timesteps, X_val.shape[1]):
            X, Y = X_val[:, (i - n_timesteps):i, :], Y_val[:, i:i + 1, :]

            self.model.evaluate(X, Y, verbose=0)

        return AutoBiLSTM(self.EpiData, self.model)

    def predict(self, target_date):
        """Predict data from fitted model

        # Argument:
            - target_date: str or datetime
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        predict_dates = []

        if target_date <= self.dates_test[-1]:

            X = np.array(np.hsplit(self.X_train.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_train[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_train[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        else:
            X = np.array(np.hsplit(self.X_test.to_numpy(), len(self.locations)))

            forecast_range = (target_date - self.dates_test[-1]).days

            for i in range(1, forecast_range + 1):
                predict_dates.append(self.dates_test[-1] + timedelta(days=i))

                Y_hat = self.model.predict(X[:, -self.n_timesteps:, :])

                Y_hat = Y_hat.reshape(Y_hat.shape[0], 1, Y_hat.shape[1])

                X = np.concatenate((X, Y_hat), axis=1)

        predict_values = list(X[:, -len(predict_dates):, 0])

        preds = pd.DataFrame(np.array(predict_values).T,
                    columns = self.Y_val.columns.values.tolist())

        preds['date'] = predict_dates

        preds = preds.melt(id_vars=['date'], \
                            var_name='location', \
                            value_name=self.target)\
                    .reindex(['location', 'date', self.target], axis=1)

        preds.date = preds.date.astype('datetime64[D]')

        return preds

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        return AutoBiLSTM(self.EpiData, self.model)

    @staticmethod
    def _build_BiLSTM_model(X, n_timesteps):
        # hyperparameters
        n_features = X.shape[-1]
        hidden_units = 64

        # functional API
        inputs = Input(shape=(n_timesteps, n_features))

        bi_lstm = layers.Bidirectional(
            layers.LSTM(hidden_units, return_sequences=True))(inputs)

        bi_lstm = layers.Bidirectional(layers.LSTM(hidden_units))(bi_lstm)

        model = layers.Dense(n_features)(bi_lstm)

        return Model(inputs, model)


def _plausible_timesteps(epi_window, max_timestep):
    """Return a list of possible numbers of timesteps to tune the model

    # Arguments:
    - epi_window: tuple or list. Contains minimum incubation period and
    maximum

    - max_timestep: the largest possible number of timesteps that data
    allows
    """

    if max_timestep > epi_window[-1]:
        return list(range(epi_window[0], max_timestep + 1)) + [max_timestep]

    else:
        return list(range(epi_window[0], max_timestep))
