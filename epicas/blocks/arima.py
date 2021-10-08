"""
Augmented Dickey–Fuller (ADF) testing for stationarity and ARIMA/SARIMA modeling
"""
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from tqdm import tqdm
import warnings
import os

from .input import InputLayer


class AutoARIMA(InputLayer):
    """
    ARIMA = AutoRegression Intergrated Moving Average

    - If training time and/or memory usage is a major concern, try VAR (Vector
    AutoRegression)
    """
    def __init__(self, EpiData, models=None, **kwargs):
        """
        # Arguments:
        - EpiData
        """

        super().__init__(EpiData, **kwargs)

        self.models = models

    def fit(self, verbose=True, **kwargs):
        """Fit train data. The procedure is given below:

        1. Find the maximum differencing order (call this tolerance)

        2. Find the best parameter d by incrementally differencing until
        time-series is stationary (based on Augmented Dickey-Fuller test)

        3. Find the best parameter p (for AR) by running stepwise tests on the
        first location

        4. Pass these two parameters to fit other locations (batches)- if there
        exist. Stepwise testing can be extremely costly if performed on all
        locations and large amount of oberservations. For the time-efficiency,
        we only run it once.

        # Argument:
            - verbose: bool. Set it to True to see ARIMA Results after fitting.
            False by default.
        """

        # TODO: multiprocessing and/or caching

        self.models = []


        Y = np.hsplit(self.Y_train.to_numpy(), len(self.locations))


        for location in tqdm(range(len(Y)), desc='ARIMA Fitting'):

            y = Y[location].flatten()

            with warnings.catch_warnings():

                warnings.filterwarnings('ignore')

                if location == 0:
                    d = self._find_diff_order(y, self.max_timestep)

                    model, p = self._stepwise_AR_variation(
                        y,
                        d,
                        self.max_timestep
                    )

                else:
                    model = ARIMA(y, order=(p, d, 0))\
                        .fit(low_memory=True, **kwargs)

            self.models.append(model)

        if verbose:
            print(self.models[0].summary())

        return AutoARIMA(self.EpiData, self.models)

    def predict(self, target_date, **kwargs):
        """Predict data from fitted models

        # Argument:
            - target_date: str or datetime
        """
        # TODO: multiprocessing

        predict_dates = []
        predict_values = []

        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        dates = self.dates_train

        last_date_trained_on = dates[-1]

        steps = (target_date - last_date_trained_on).days

        start = len(dates)

        end = start + steps - 1

        predict_dates = [last_date_trained_on + timedelta(days=1 + x) for
                              x in range(steps)]

        for location in tqdm(range(len(self.models)), desc='ARIMA Predicting'):

            forecast = self.models[location].predict(start, end)

            predict_values.append(forecast)

        preds = pd.DataFrame(np.array(predict_values).T,
                            columns = self.Y_train.columns.values.tolist())

        preds['date'] = predict_dates

        preds = preds.melt(id_vars=['date'], \
                            var_name='location', \
                            value_name=self.target)\
                    .reindex(['location', 'date', self.target], axis=1)

        preds.date = preds.date.astype('datetime64[D]')

        return preds

    def save(self, path, **kwargs):
        """Export fitted models to specific path for later use

        # Argument:
            - path: str. the name of the directory to export ARIMAResults
            instances into.
        """
        if not os.path.exists(path):
            os.mkdir(path)

        for i in tqdm(range(len(self.models)), desc='Saving models...'):
            self.models[i].save(path + f'/{self.locations[i]}.pkl')

    def load(self, path):
        fs = os.listdir(path + '/')
        models = []

        for f in fs:
            models.append(ARIMAResults.load(path + '/' + f))

        return AutoARIMA(self.EpiData, models)

    @staticmethod
    def _adf(y):
        """
        Implement Augmented Dickey–Fuller to test for stationarity of series

        # Output:
            - If True, series may be stationary. Otherwise, it may be
            non-stationary

        # References:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools
        .adfuller.html

        MacKinnon, J.G. 1994. “Approximate asymptotic distribution functions for
        unit-root and cointegration tests. Journal of Business and Economic
        Statistics 12, 167-76.
        """
        adf = adfuller(y)

        if all(adf[0] > val for val in list(adf[4].values())):
            return False  # series may be non-stationary

        else:
            return True  # series may be stationary

    @staticmethod
    def _find_diff_order(y, max_order):
        for d in range(0, max_order + 1):
            if AutoARIMA._adf(np.diff(y, d)):
                return d

    @staticmethod
    def _stepwise_AR_variation(y, d, p_end, p_start=0):
        # TODO: EarlyStopping mechanism
        """Choosing the optimal parameter using AIC and BIC metrics

        # Arguments:
            - y: array. the series

            - dates: array of datetime

            - d: differencing order (obtained from ADF)

            - p_start: int. First p to try with.

            - p_end: int. maximum p

            - early_stopping_delta: float or int. Inspired by EarlyStopping
            callback in Tensorflow, stepwise testing will be stopped if AIC and
            BIC improvements fall under threshold% (i.e., 0.1 for 10%) of the
            previous ones. This helps save time and computational cost, while
            mitigating overfitting.
        """
        optimal_aic = None
        optimal_bic = None
        optimal_model = None
        optimal_p = 0

        for p in range(p_start, p_end + 1):
            model = ARIMA(y, order=(p, d, 0)).fit()

            if (optimal_aic, optimal_bic) == (None, None):
                optimal_aic, optimal_bic, optimal_p = model.aic, model.bic, p
                optimal_model = model

            elif model.aic < optimal_aic and model.bic < optimal_bic:

                optimal_aic, optimal_bic, optimal_p = model.aic, model.bic, p
                optimal_model = model

        return optimal_model, optimal_p
