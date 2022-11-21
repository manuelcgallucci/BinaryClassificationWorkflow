# Import libraries
from pstats import Stats
import numpy as np
import pandas as pd

def import_data(csv_path, col_names, true_label_str=(np.Nan,np.Nan),normalize="std", class_label="last", dummy_cols=None, replace_nan="mean"):
    """
    INPUT:
    -   csv_path: path to the data in csv format
    -   col_names: names of the columns of the dataset
    -   true_label_str: tuple of the classification values (true_label,false_label)
    -   normalize: Normalization type for all columns. ("std", "0_1")
    -   dummy_cols: names of the columns to apply dummy variables
    -   replace_nan: Way to replace nan or missing values. ("mean", "median", "remove")

    OUTPUT: 
    -   X: DataFrame of size (n_samples x n_features) with column names included.
    -   y: DataFrame of size (n_samples x 1) that represents the labels
    """

    # Auxiliar function
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
    rows_to_delete = []
    for i in range(np.size(y)):
        if (y[i]=="" or y[i]==" " or y[i]=='?' or y[i]==np.Nan):
            rows_to_delete.append(i)
    for a in rows_to_delete:
        y = np.delete(y,a)
        X = np.delete(X,a, axis=0)

    # Turn labels to integers 0 or 1, supposing every label has a valid value and of the same dtype
    rows_to_delete = []
    if isinstance(y[0], str):
        if true_label_str != (np.Nan,np.Nan):
            for i in range(np.size(y)):
                if y[i]==true_label_str[0]:
                    # if value is true
                    y[i]=1
                elif y[i]==true_label_str[1]:
                    y[i]=0
                else:
                    rows_to_delete.append(i)
            for a in rows_to_delete:
                y = np.delete(y,a)
                X = np.delete(X,a, axis=0)
        else:
            # Here we suppose the data was evaluated and there are no missing labels
            y = (y==y[0]).astype(int)
    elif isinstance(y[0],int):
        for i in range(np.size(y)):
            assert (y[i] == 1 or y[i] == 0), 'Label values are not 0 or 1'
    elif isinstance(y[0],bool):
        y = y.astype(int)


    # Call to function clean_data
    clean_data(X, replace_nan=replace_nan)

    return data
