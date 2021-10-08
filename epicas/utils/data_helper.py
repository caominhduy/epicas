import pandas as pd
import json

def get_format(path):
    return path.split('.')[-1].lower()


def count_na(df):
    nulls = df.isnull().sum()

    for column in df.columns:
        print(f'NaN / NaT data in {column:30}:' +
            f'{(nulls/len(df)*100)[column].round(1):>5} %')

    return nulls


def read_long_csv(file_input, labels, start_date, tot_pop, dropna, **kwargs):
    col_map = {x: y for y, x in labels.items()}

    df = pd.read_csv(
        file_input,
        usecols=list(labels.values()),
        **kwargs)

    if dropna == True:
        df = df.dropna()
    elif type(dropna) == list:
        df = df.dropna(subset=dropna)

    df = df.rename(columns=col_map)

    if 'tot_pop' not in list(df.columns.values):
        if tot_pop != None:
            df['tot_pop'] = tot_pop

    return df


def read_wide_csv(file_input, labels, start_date, tot_pop, dropna, dropcols,
                wide_label, **kwargs):
    col_map = {x: y for y, x in labels.items()}

    df = pd.read_csv(file_input, **kwargs)

    if dropna == True:
        df = df.dropna()
    elif type(dropna) == list:
        df = df.dropna(subset=dropna)

    if type(dropcols) == list:
        df = df.drop(columns=dropcols)

    df = df.rename(columns=col_map)

    cols = list(labels.keys())

    if 'location' in labels:
        cols.remove('location')

        df = df.melt(
            id_vars=['location'],
            var_name = 'date',
            value_name = wide_label
        )


    else:
        cols.remove('date')

        df = df.melt(
            var_name = 'date',
            value_name = wide_label
        )

    return df


def variables_merge(static_1, time_series_1, static_2, time_series_2):
    if static_1 != None and static_2 != None:
        static = list(set(static_1 + static_2))

    elif static_1 != None:
        static = static_1

    elif static_2 != None:
        static = static_2

    else:
        static = None

    if time_series_1 != None and time_series_2 != None:
        time_series = list(set(time_series_1 + time_series_2))

    elif time_series_1 != None:
        time_series = time_series_1

    elif time_series_2 != None:
        time_series = time_series_2

    else:
        time_series = None

    return static, time_series
