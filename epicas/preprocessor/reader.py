"""
This module reads structured data (CSV, Excel, JSON, etc.) to Pandas dataframes
"""
import pandas as pd

from epicas import utils

class StructuredData:
    """
    # Arguments:
        file_input = str. Path or URL to the dataset.

        date = str or None.
            - Specify the name of column that will be used as date column

            - If None (default), data will be parsed as static data. Static data
            is the data that is not changed, presumed to be the same through out
            outbreaking season. e.g., population data, asthma mortality.

        location = str or None.
            - Specify the name of the location column in your data.
            - If None, dataset will be read as single location.

        incidence = str or None.
            - If specified, the name of the column that will be read as incident
            cases (new cases over an interval of days)

        prevalence = str or None.
            - If specified, the name of the column that will be read as prevalent
            cases (currently active cases at some specific time period)

        death = str or None.
            - If specified, the name of the column that will be read as deaths due
            to infection

        recovered = str or None.
            - If specified, the name of the column that will be read as recovered
            cases

        cumulative = bool. False by default.
            - If True, the values of incident, prevalent, recovered cases, and/or
            deaths will be read as "total recorded from start of outbreak to
            specified date."

        df = pandas.DataFrame

        **kwargs: other arguments can be passed into pandas.read_csv
            (see pandas documentation for all possible arguments).
    """

    def __init__(self, file_input = None, date = None, location = None,
                incidence = None, prevalence = None, death = None,
                recovered = None, cumulative = False, df = None,
                static_vars = None, time_series_vars=None, **kwargs):

        self.path = file_input
        self.date = date
        self.location = location
        self.incidence = incidence
        self.prevalence = prevalence
        self.death = death
        self.recovered = recovered
        self.cumulative = cumulative
        self.variables = {}
        self.variables['static'] = static_vars
        self.variables['time_series'] = time_series_vars

        headers = {
            self.date : 'date',
            self.location : 'location',
            self.incidence : 'incidence',
            self.prevalence : 'prevalence',
            self.recovered : 'recovered',
            self.death : 'death'
        }

        del headers[None]

        if isinstance(df, pd.DataFrame):
            self.df = df.rename(columns=headers)
            if 'date' in self.df.columns.values.tolist():
                self.df.date = pd.to_datetime(self.df.date, errors='coerce')

        else:
            self.df = pd.read_csv(self.path, **kwargs)
            self.df = self.df.rename(columns=headers)
            if 'date' in self.df.columns.values.tolist():
                self.df.date = pd.to_datetime(self.df.date, errors='coerce')

        if self.variables['static'] == None and self.date == None:
            self.variables['static'] = self.df.columns.values.tolist()

        elif self.variables['time_series'] == None:
            self.variables['time_series'] = self.df.columns.values.tolist()


    def __add__(self, next):
        """
        # Usage:
            df = StructuredData(...)
            df2 = StructuredData(...)
            df = df + df2
        """
        commons = list(self.df.columns.intersection(next.df.columns))

        joined = self.df.merge(next.df, on=commons)

        static_vars, time_series_vars = utils.variables_merge(
            self.variables['static'],
            self.variables['time_series'],
            next.variables['static'],
            next.variables['time_series']
        )

        return StructuredData(
            df = joined,
            static_vars = static_vars,
            time_series_vars = time_series_vars)


    def __str__(self):
        return f'StructuredData{list(self.df.columns)}\n\n' + \
                f'Variables: {self.variables}\n\n' + \
                f'{self.df}\n\n'


    def stats(self):
        """Examine StructuredData for NaN"""

        utils.count_na(self.df)


    def drop(self, drop_list):
        """Drop features from StructuredData

        # Argument:
            - drop_list: list. List of features to be dropped"""

        self.df = self.df.drop(columns=drop_list)


class UnstructuredData:
    """ To be developed """

    def __init__(**kwargs):
        pass
