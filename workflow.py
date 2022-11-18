# Import libraries
from pstats import Stats
import numpy as np
import pandas as pd

def import_data(csv_path, col_names, normalize="std", class_label="last", dummy_cols=None, replace_nan="mean"):
    """
    INPUT:
    -   csv_path: path to the data in csv format
    -   col_names: names of the columns of the dataset
    -   normalize: Normalization type for all columns. ("std", "0_1")
    -   dummy_cols: names of the columns to apply dummy variables
    -   replace_nan: Way to replace nan or missing values. ("mean", "median", "remove")

    OUTPUT: 
    -   DataFrame of size (n_samples x n_features) with column names included.
    """

    # Read data
    data = pd.read_csv(csv_path)

    # Drop label (classification) column
    if (class_label=="last"):
        label_to_drop = data.columns[-1]
    else:
        label_to_drop = class_label
    X = data.drop(label_to_drop, axis=1).values
    y = data[label_to_drop].values

    # Remove empty labels


    # Turn labels to integers 0 or 1, supposing every label has a valid value and of the same dtype
    if isinstance(y[0], str):
        y = pd.get_dummies(y, drop_first=True)
    else if isinstance(y[0],int):

    else if isinstance(y[0],bool):



    labels = data.drop()






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

    return data