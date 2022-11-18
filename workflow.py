from pstats import Stats
import numpy as np
import pandas as pd


def import_data(csv_path, col_names, normalize="std", dummy_cols=None, replace_nan="mean"):
    """
    csv_path: path to the data in csv format
    col_names: names of the columns of the dataset
    normalize: Normalization type for all columns. ("std", "0_1")\
    dummy_cols: names of the columns to apply dummy variables
    replace_nan: Way to replace nan or missing values. ("mean", "median", "remove")
    """


    return data


def clean_data(data, replace_nan="mean"):
    """
    data: data to be cleaned
    replace_nan: Way to replace nan or missing values. ("mean", "median", "remove")
    """
    if replace_nan == "mean":
        data = data.fillna(data.mean())
    elif replace_nan == "median":  
        data = data.fillna(data.median())
    elif replace_nan == "remove":
        data = data.dropna()

    # Removing rows with outliers
    data = data[(np.abs(Stats.zscore(data)) < 3).all(axis=1)]

    # Normalizing data
    data = (data - data.mean()) / data.std()

    # Shuffling data
    data = data.sample(frac=1).reset_index(drop=True)

    return data