"""
This module performs feature engineering on StructuredData
"""

import pandas as pd
import math

from epicas import preprocessor, utils, settings

class EpiData:
    def __init__(self, StructuredData, y, df=None, disease='generic', **kwargs):
        """
        # Arguments:

            - StructuredData

            - y = str. This is your target variable. Choose either:
                + 'incidence'
                + 'prevalence'
                + 'recovered'
                + 'death'

            - disease = str. If unspecified, 'generic' is passed. If disease is
            unknown or in doubt, use generic options:
                + 'generic'
                + 'generic_aerosol'
                + 'generic_body_fluid'
                + 'generic_fecal_oral'
                + 'generic_respiratory'
                + 'generic_respiratory_droplet'

            or choose from the list of well-known based on similarities:
                + 'influenza'
                + 'covid19'
                + 'covid19_alpha'
                + 'covid19_delta'
                + 'sars'
                + 'mers'
                + 'common_cold'
                + 'ebola'
                + 'measles'
                + 'mump'
                + 'hiv'
                + 'hantavirus'
                + 'polio'
                + 'chickenpox'
        """

        self.StructuredData = StructuredData

        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            self.df = self.StructuredData.df

        self._resort(self.df)
        self.target = y
        self.disease = disease

        if 'location' not in self.df.columns.values.tolist():
            self.df.location = 'single'

        self.headers = StructuredData.df.columns.values.tolist()
        self.features = self._list_of_features(self.headers, self.target)
        self.epi_params = settings.static_params()['infectious_info']

        self._validate(self.headers, self.disease, self.epi_params)


    def __str__(self):
        return f'EpiData{list(self.df.columns)}\n\n' + \
                f'{self.df}\n\n'


    def corr(self, **kwargs):
        """ Calculate correlation between explanatory variables and target
        variables
        """

        self._corr(self.df, self.headers, self.target, self.features, **kwargs)


    def stats(self):
        """ Examine EpiData for NaNs """

        utils.count_na(self.df)


    def lag_reduction(self, subset=None, mid_point=0, sliding_window=None,
        verbose=True):
        """ Shift time-series bi-directionally to maximize correlation between
        predictor and target

        # Arguments:

            - subset: list-like or callable. List of explanatory variables
            whose momentums are presumed to precede target variable.

            - mid_point: int. The midpoint of sliding window

            - sliding_window: int or None. Window size (days) to try shifting
            time-series between. If None, the window will be inferred from
            usual incubation periods for specified disease.

            - verbose: bool. set to False to stop reporting shifted orders
        """

        if sliding_window == None:
            sliding_window = \
                self.epi_params[self.disease]['incubation_period'][-1]

        self.df = self._find_optimal_shift(
            subset,
            self.df,
            self.target,
            mid_point,
            sliding_window,
            verbose
        )

        return EpiData(self.StructuredData, self.target, self.df)


    def imputation(self, method='median', **kwargs):
        """ Fill NaN values

        # Arguments:

            - method: str. Choose either:

                + 'median': fill missing values with medians, this option is
                more robust to outliers.

                + 'mean': fill missing values with mean values

                + 'zero': fill missing values with 0

                + 'ffill': fill missing values by propagating last valid
                observations forward to next

                + 'bfill': fill missing values with next available observations
        """
        if method == 'median':
            for var in self.headers:
                if var not in ['date', 'location']:
                    self.df[var] = self.df[var].fillna(self.df[var].median())

        elif method == 'median':
            for var in self.headers:
                if var not in ['date', 'location']:
                    self.df[var] = self.df[var].fillna(self.df[var].mean())

        elif method == 'zero':
            for var in self.headers:
                if var not in ['date', 'location']:
                    self.df[var] = self.df[var].fillna(0)

        elif method == 'ffill':
            for var in self.headers:
                if var not in ['date', 'location']:
                    self.df[var] = self.df[var].fillna(method='ffill')

        elif method == 'bfill':
            for var in self.headers:
                if var not in ['date', 'location']:
                    self.df[var] = self.df[var].fillna(method='bfill')

        else:
            raise ValueError(f'{method} is not a built-in method. See documentation.')

        return EpiData(self.StructuredData, self.target, self.df)


    def feature_selection(self, n, **kwargs):
        # TODO: implement SHAP model explainer and/or LIME

        """ Perform feature selection and retain best n features.

        Take this function with a grain of salt: unless your task is to quickly
        dive into modeling, this is bad as it may not correctly interpret the
        behavior of our model architecture.

        # Arguments:

            - n: int or float. Number of best features that you want to keep.

                + If <1: proportional to total number of features

                + If >=1: number of features to be kept
        """

        max_n = len(self.features)

        if n > max_n or n<=0:
            raise ValueError('Number of features is invalid')

        elif n < 1:
            n = math.ceil(max_n * n)


        correlations = _corr(
            self.df,
            self.headers,
            self.target,
            self.features,
            verbose=False
        )

        top_features = sorted(
            correlations,
            key = correlations.get,
            reverse=True
        )

        top_n_features = top_features[:n]

        for feature in self.features:
            if feature not in top_n_features:
                self.df = self.df.drop(columns = feature)
                if feature in self.StructuredData.variables['static']:
                    self.StructuredData.variables['static'].remove(feature)
                if feature in self.StructuredData.variables['time_series']:
                    self.StructuredData.variables['time_series'].remove(feature)

        return EpiData(self.StructuredData, self.target, self.df)


    def normalization(self, range=[0, 1], subset='all'):
        """ Apply normalization to variables (similar to sklearn.preprocessing
        .MinMaxScaler)

        # Arguments:

            - range: list-like or callable. The range of features to be
            normalized into. Default: 0<=x<=1.

            - subset = str or list-like (callable). The subset of columns
            to be normalized

                + If 'all': all variables will be normalized except target
                variable

                + If 'time-series': all time-series variables will be normalized
                except target variable

                + If 'static': all static variables will be normalized

                + If list-like: list of specific features to be normalized
        """

        excl = ['location', 'date', self.target]
        norm_variables = []

        if subset == 'all':
            norm_variables = self.features

        if subset == 'time_series':
            for var in self.StructuredData.variables['time_series']:
                if var not in excl:
                    norm_variables.append(var)

        if subset == 'static':
            for var in self.StructuredData.variables['static']:
                if var not in excl:
                    norm_variables.append(var)

        elif type(subset) == list:
            norm_variables = subset

        for var in norm_variables:
            x = self.df[var]
            stdev = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
            self.df[var] = stdev * (range[-1] - range[0]) + range[0]

        return EpiData(self.StructuredData, self.target, self.df)


    def outlier_removal(self, variables, upper=None, lower=None, **kwargs):
        """ Clip outliers off EpiData based on set thresholds

        # Arguments:

            - upper: dict or None. Map variable to its upper threshold

            - lower: dict or None. Map variable to its upper threshold
        """
        self._clip(self.df, variables, upper, lower, **kwargs)

        return EpiData(self.StructuredData, self.target, self.df)


    def cumulative_to_incident(self, subset='target', **kwargs):
        """ Convert cumulative records to incident observations

        # Arguments:

            - subset: str or list-like (callable). Choose either:

                + 'target': only convert target variable

                + 'all': convert all variables that are time-series

                + List of specific columns to be converted
        """

        if self.StructuredData.cumulative:

            if subset == 'all':
                for var in self.variables['time_series']:
                    if var not in ['location', 'date']:
                        df[var] = df[var].diff(**kwargs)

            elif subset == 'target':
                df[self.target] = df[self.target].diff(**kwargs)

            else:
                for var in subset:
                    df[var] = df[var].diff(**kwargs)

        return EpiData(self.StructuredData, self.target, self.df)


    def target_to_ma(self, window, **kwargs):
        """ Convert target observations to their moving average (MA) values

        # Arguments:

            - window: int. Fixed window size to calculate rolling average.
        """

        self.df[self.target] = self.df[self.target].rolling(window).mean()

        self.df[self.target] = self.df[self.target].fillna(method='bfill')

        return EpiData(self.StructuredData, self.target, self.df)


    @staticmethod
    def _corr(df, headers, target, features, verbose = True, **kwargs):
        var_tuples = []
        pearson_scores = []
        correlation_table = {}

        for var in features:

                var_tuples.append(f'({var}, {target})')
                score = df[var].corr(df[target], **kwargs)
                pearson_scores.append(score.round(5))

                correlation_table[var] = score

        if verbose:

            for tuple, score in zip(var_tuples, pearson_scores):
                print(f'Correlation of {tuple:40}:{score:>10}')

        return correlation_table


    @staticmethod
    def _validate(headers, disease, params):
        if 'date' not in headers:
            raise ValueError('Date is missing from your data')

        if not any(var in ['incidence', 'prevalence', 'recovered', 'death']
                   for var in headers):

            raise ValueError('Need at least one target value for EpiData')

        if disease not in params:
            raise ValueError(f'Invalid disease argument: "{disease}"')


    @staticmethod
    def _clip(df, upper, lower):
        if upper != None:
            for var in list(upper.keys()):
                df[var].clip(upper=upper[var], inplace=True)

        if lower != None:
            for var in list(lower.keys()):
                df[var].clip(lower=lower[var], inplace=True)


    @staticmethod
    def _list_of_features(headers, target):
        features = []

        for feature in headers:

            if feature not in ['location', 'date', target]:

                features.append(feature)

        return features


    @staticmethod
    def _find_optimal_shift(subset, df, target, mid_point, sliding_window,
                            verbose):
        start = mid_point - int(sliding_window/2)
        end = start + sliding_window
        y = df[target]

        for var in subset:

            best_score = df[var].corr(y)
            best_shift = 0

            for shift in range(start, end):

                shifted = df.groupby('location')[var].shift(shift)
                score = abs(shifted.corr(y))

                if score > best_score:
                    best_score = score
                    best_shift = shift

            if verbose:
                print(f'Optimal shift for {var:<10}:{best_shift:>3}')

            shifted = df.groupby('location')[var].shift(best_shift)
            df = df.drop(columns=[var])
            df[var] = shifted

        return df.dropna(subset=subset).reset_index(drop=True)

    @staticmethod
    def _resort(df):
        return df.sort_values(['location', 'date'], inplace=True)
