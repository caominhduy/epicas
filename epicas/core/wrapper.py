"""
Ensemble forecasts from models.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime

from epicas.blocks.input import InputLayer
from epicas.blocks.arima import AutoARIMA
from epicas.blocks.sequence import AutoLSTM, AutoBiLSTM, AutoGRU
from epicas.blocks.attention import AutoAttention


class Ensemble(InputLayer):
    def __init__(self, EpiData, models, target_date):
        """Automatically train models, validates and gets weights for weighted
        ensemble model

        # Arguments:
            - EpiData: feature engineered models

            - models: list. models to use, choose from:

                + 'ARIMA'

                + 'attention'

                + 'LSTM'

                + 'BiLSTM'

                + 'GRU'

            - target_date: str or datetime
        """
        super().__init__(EpiData)

        self.df = self.Y_out.reset_index(drop=False)\
                            .melt(id_vars=['date'],
                                  var_name='location',
                                  value_name=self.target)\
                            .reindex(['location', 'date', self.target], axis=1)

        self.df = self._unweighted_ensemble(
            models,
            self.df,
            EpiData,
            target_date,
            self.target
        )

    def get_predict(self, in_sample=True):
        """Return a Pandas DataFrame of predicted values

        # Arguments:
            - in_sample: bool. If True: including in-sample grouth truth data
            in the final EpiData. Otherwise, only outputting predicted data

        """
        if in_sample:
            return self._in_sample(self.df, self.target)
        else:
            return self._out_of_sample(self.df, self.target)

    def save_predict(self, path, in_sample=True):
        """Return a Pandas DataFrame of predicted values

        # Arguments:
            -path: str. Export dataset as CSV.

            - in_sample: bool. If True: including in-sample grouth truth data
            in the final EpiData. Otherwise, only outputting predicted data

        """
        if in_sample:
            df = self._in_sample(self.df, self.target)
        else:
            df = self._out_of_sample(self.df, self.target)

        df.to_csv(path, index=False)

    def __str__(self):
        return f'{self.df}'

    @staticmethod
    def _run_model(model_name, EpiData, date):

        if model_name.lower() == 'attention':
            output = AutoAttention(EpiData).fit().predict(date)

        elif model_name.lower() == 'arima':
            output = AutoARIMA(EpiData).fit().predict(date)

        elif model_name.lower() == 'lstm':
            output = AutoLSTM(EpiData).fit().predict(date)

        elif model_name.lower() == 'bilstm' or model_name.lower() == 'bi-lstm':
            output = AutoBiLSTM(EpiData).fit().predict(date)

        elif model_name.lower() == 'gru':
            output = AutoGRU(EpiData).fit().predict(date)

        else:
            raise ValueError(f'Invalid model name: {model_name}')

        return output

    @staticmethod
    def _unweighted_ensemble(model_names, df, EpiData, date, target):

        for model_name in model_names:

            output = Ensemble._run_model(model_name, EpiData, date)
            output = output.rename(columns={target: model_name})

            df = df.merge(output, on=['date', 'location'], how='outer')

        df[f'{target}_preds'] = df[model_names].mean(axis=1)

        df = df.drop(columns=model_names)

        return df

    @staticmethod
    def _out_of_sample(df, target):

        df = df[df[target].isnull()]
        df = df.drop(columns=[target])

        df[target] = df[f'{target}_preds']
        
        df = df.drop(columns=[f'{target}_preds'])

        return df

    @staticmethod
    def _in_sample(df, target):

        df[target] = df[target].fillna(df[f'{target}_preds'])
        df = df.drop(columns=[f'{target}_preds'])

        return df
