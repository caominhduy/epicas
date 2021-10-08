"""
Encode EpiData to input layer
"""

import warnings
import numpy as np
from datetime import datetime


class InputLayer:
    def __init__(self, EpiData, **kwargs):
        # TODO: add arg for custom train-val-test ratio
        """
        # Arguments:

            - EpiData: EpiData obj. The feature engineered data

            - StructuredData: StructuredData obj. Original data

            - target = int. Target variable

            - target_date = int, str, or datetime.

                + If int: the number of days into the future that is relative
                to the last observation.

                + If str: isoformat string to be parsed as date (for example,
                '2021-10-02' is a valid string for Oct 2, 2021)

                + If datetime: this date will be read directly without parsed

            - train_test_ratio = None or float.

                + If float: (between 0 and 1) this will be the ratio of the
                training data proportionally to the EpiData. The rest will be
                splited 50/50 into validation and testing

                + If None: the ratio will be automatically scaled based on
                the size of the EpiData

        """
        self.EpiData = EpiData
        self.StructuredData = self.EpiData.StructuredData
        self.target = self.EpiData.target
        self.disease = self.EpiData.disease
        self.windows = self.EpiData.epi_params[self.disease]['incubation_period']
        self.features = self.EpiData.features
        self.df = self._multi_indexing(self.EpiData.df)

        self.time_series_vars = self._update_variable_list(
            self.EpiData.StructuredData.variables['time_series'],
            self.df,
            self.target
        )

        self.static_vars = self._update_variable_list(
            self.EpiData.StructuredData.variables['static'],
            self.df,
            self.target
        )

        self.X, self.X_train, self.X_val, self.X_test, self.Y,\
            self.Y_train, self.Y_val, self.Y_test, self.dates_train,\
            self.dates_val, self.dates_test, self.max_timestep,\
            self.locations, self.Y_out =\
            self._train_test_split(
                self.df,
                self.target,
                self.windows[-1],
                self.time_series_vars,
                self.static_vars
            )

    def __str__(self):
        return f'EpiData{list(self.df.columns)}\n\n' + \
            f'{self.df}\n\n'

    @staticmethod
    def _infer_ratio(len, max_window):
        """Return split sizes based on length of available data"""

        if len >= 100000:  # training: 99%, validation: 0.5%, testing: 0.5%
            train_cut = int(0.99 * len)
            validation_cut = int(0.995 * len)

        elif len >= 10000:  # training: 90%, validation: 5%, testing: 5%
            train_cut = int(0.9 * len)
            validation_cut = int(0.95 * len)

        else:  # training: 80%, validation: 10%, testing: 10%
            train_cut = int(0.8 * len)
            validation_cut = int(0.9 * len)

        if (len - validation_cut + 1) < max_window:
            train_cut -= max_window - (len - validation_cut + 1)
            validation_cut -= max_window - (len - validation_cut + 1)

        if (validation_cut - train_cut + 1) < max_window:
            compensation = max_window - (validation_cut - train_cut + 1)
            train_cut -= compensation
            validation_cut -= compensation

        if train_cut < max_window or validation_cut < 0:
            return 0, 0
        else:
            return train_cut, validation_cut

    @staticmethod
    def _train_test_split(df, target, max_window, time_series_vars,
                          static_vars):
        """Split dataset into training, validating, and testing NumPy arrays

        # Arguments:
            - df: DataFrame fetched from EpiData

            - target: name of target variable

            - min_window: the minimum number of days to look back

            - max_window: the maximum number of days to look back

            - time_series_vars: list of time_series_variables (except 'date',
            'location' and target variable)

            - static_vars: list of time_series_variables (except 'location')

        # Outputs:
            - train_data: list of (dates, target values, time-series regressors,
            static regressors) to be used for training, one for each location

            - val_data: list of (dates, target values, time-series regressors,
            static regressors) to be used for validation, one for each location

            - test_data: list of (dates, target values, time-series regressors,
            static regressors) to be used for testing, one for each location

        """

        locations = list(set(df.index.values.tolist()))

        X = df.pivot_table(
                index='date',
                columns='location',
                values=[target] + time_series_vars
            ).ffill().bfill()

        dates = X.index.values.astype('datetime64[D]').astype(datetime)
        length = len(dates)

        X = X\
            .reset_index(drop=True)\
            .swaplevel(axis=1)\
            .sort_index(axis=1)\
            .reindex([target] + time_series_vars, axis=1, level=1)

        X_train = X[:int(length/1.25)]
        dates_train = dates[:int(length/1.25)]

        X_val = X[int(length/1.25):int(length/(10/9))]
        dates_val = dates[int(length/1.25):int(length/(10/9))]

        X_test = X[int(length/(10/9)):]
        dates_test = dates[int(length/(10/9)):]

        Y_out = df.pivot_table(
                index='date',
                columns='location',
                values=target
            ).ffill().bfill()

        Y = Y_out.reset_index(drop=True)

        Y_train = Y[:int(length/1.25)]

        Y_val = Y[int(length/1.25):int(length/(10/9))]

        Y_test = Y[int(length/(10/9)):]

        max_timestep = min(max_window, len(dates_val), len(dates_test))

        return X, X_train, X_val, X_test, Y, Y_train, Y_val, Y_test,\
            dates_train, dates_val, dates_test, max_timestep, locations, Y_out

    @staticmethod
    def _multi_indexing(df):
        return df.set_index('location')

    @staticmethod
    def _find_available_length(df, locations):

        if len(locations) != 1:
            max_count = 0

            for location in locations:
                new_count = df.loc[location].shape[0]

                if new_count > max_count:
                    max_count = new_count

        else:
            max_count = df.loc[location[0]].shape[0]

        return max_count

    @staticmethod
    def _update_variable_list(li, df, target):

        cols = []

        for column in li:
            if column in df.columns.values.tolist() and \
                    column not in [target, 'date', 'location']:

                cols.append(column)

        return cols
