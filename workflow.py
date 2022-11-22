# Import libraries
from pstats import Stats
import numpy as np
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from sklearn.utils import shuffle

import seaborn as sns
import matplotlib.pyplot as plt

import os 

def import_data(csv_path, col_names, true_label_str=None,normalize="std", class_label="last", dummy_cols=None, replace_nan=True, random_state=42):
    """
    INPUT:
    -   csv_path: path to the data in csv format
    -   col_names: names of the columns of the dataset
    -   true_label_str: tuple of the classification values (true_label,false_label)
    -   normalize: Normalization type for all columns. ("std", "0_1")
    -   dummy_cols: names of the columns to apply dummy variables
    -   replace_nan: Way to replace nan or missing values. Mode for ints and Mean for floats 
    OUTPUT: 
    -   data: DataFrame of size (n_samples x n_features) with column names included.
    -   y: DataFrame of size (n_samples x 1) that represents the labels
    """
    if true_label_str is None:
        true_label_str=(np.nan,np.nan)

    # Read data TODO agregar columnas nombres 
    data = pd.read_csv(csv_path)
    
    # Drop label (classification) column
    if (class_label=="last"):
        label_to_drop = data.columns[-1]
    else:
        label_to_drop = class_label
    X = data.drop(label_to_drop, axis=1)
    y = data[label_to_drop]

    # Fill dummy values with the mode
    if dummy_cols is not None:
        for col in dummy_cols: 
            X[col].fillna(X[col].mode(), inplace=True)
    
    # Dummy values 
    # Replaces the column names with dummys 
    X = pd.get_dummies(X, columns=dummy_cols, drop_first=True)

    # Remove empty labels
    rows_to_delete = []
    for i in range(np.size(y)):
        if (y[i]=="" or y[i]==" " or y[i]=='?' or y[i]==np.nan):
            rows_to_delete.append(i)
    y = y.drop(rows_to_delete, axis=0)
    X = X.drop(rows_to_delete, axis=0)

    # Turn labels to integers 0 or 1, supposing every label has a valid value and of the same dtype
    rows_to_delete = []
    if isinstance(y[0], str):
        if true_label_str != (np.nan,np.nan):
            for i in range(np.size(y)):
                if y[i]==true_label_str[0]:
                    # if value is true
                    y[i]=1
                elif y[i]==true_label_str[1]:
                    y[i]=0
                else:
                    rows_to_delete.append(i)
            y = y.drop(rows_to_delete, axis=0)
            X = X.drop(rows_to_delete, axis=0)
        else:
            # Here we suppose the data was evaluated and there are no missing labels
            y = (y==y[0]).astype(int)
    elif isinstance(y[0],int):
        for i in range(np.size(y)):
            assert (y[i] == 1 or y[i] == 0), 'Label values are not 0 or 1'
    elif isinstance(y[0],bool):
        y = y.astype(int)

    # Dealing with missing values
    # Checking dummy columns to replace nan median
    if replace_nan:
        # TODO
        # X column type 
        for col in X.columns:
            X[col].fillna(X[col].mean(), inplace=True)
    else:
        # TODO 
        # Remove nan rows
        pass
    # Normalizing data
    if normalize == "std":
        X = (X - X.mean()) / X.std()
    elif normalize == "0_1":
        X = (X - X.min()) / (X.max() - X.min())
    
    # Shuffling data
    if random_state is not None:
        X, y = shuffle(X, y, random_state=random_state)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

    return X, y.values

def train_model(model, data, labels, test_size, random_state=42, plot_results=None, cross_validation=None):
    """
    model: The model to train from scikit learn. Must accept the functions .fit(x_train, y_train)
    data: Data to train and test on 
    test_size: Percentage of data to use on the test split
    plot_results: Path to plot the results plots, if None it doesnt plot
        * If not cross_validation : plots the confusion matrix
        * If     cross_validation : plots the average confusion matrix and all the metric comparison for each cross validation
    cross_validation: Number of cross validations to perform, 1 if its None

    returns the accuracy, precision and recall of the model as a tuple and the confusion matrix
    IF cross validation is used, then these are return in the form of lists
    """
    assert cross_validation >= 1, "Error:Cross validation should be bigger or equal to one"
    
    if cross_validation is None:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        cf_matrix = confusion_matrix(y_pred, y_test)

        if plot_results is not None:    
            if not os.path.exists(plot_results):
                os.mkdir(plot_results)
            if not os.path.exists(os.path.join(plot_results, "results")):
                os.mkdir(os.path.join(plot_results, "results"))
            fig, ax = plt.subplots()
            ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True values')
            ax.set_xlabel('Predicted values')
            fig = ax.get_figure()
            fig.savefig(os.path.join(plot_results, "results", "confusion_matrix.png")) 
    else:
        # Perform cross validation
        cross_val = ShuffleSplit(n_splits=cross_validation, test_size=test_size, random_state=random_state)
        
        accuracy = []
        precision = []
        recall = []
        cf_matrix = []

        for train_index, test_index in cross_val.split(data):
            # Copy the model so it can be trained from 0 each time
            model_ = clone(model)
            X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], labels[train_index], labels[test_index]
            # Fit and predict the model
            model_.fit(X_train, y_train)
            y_pred = model_.predict(X_test)
            # Save the parameters of the cross validation
            accuracy.append(accuracy_score(y_pred, y_test))
            precision.append(precision_score(y_pred, y_test))
            recall.append(recall_score(y_pred, y_test))
            cf_matrix.append(confusion_matrix(y_pred, y_test))

        if plot_results is not None:    
            if not os.path.exists(plot_results):
                os.mkdir(plot_results)
            if not os.path.exists(os.path.join(plot_results, "results_crval")):
                os.mkdir(os.path.join(plot_results, "results_crval"))
            # Plot the mean confusion matrix
            fig, ax = plt.subplots()
            ax = sns.heatmap(np.mean(np.array(cf_matrix), axis=0), annot=True,cmap='Blues', fmt='g')
            ax.set_title('Mean Confusion Matrix')
            ax.set_ylabel('True values')
            ax.set_xlabel('Predicted values')
            fig = ax.get_figure()
            fig.savefig(os.path.join(plot_results, "results_crval", "confusion_matrix.png"))

            plt.figure()
            x_axis = np.linspace(0, cross_validation, cross_validation, endpoint=False)
            # Plot the metric comparison
            raw_metrics = {"accuracy":accuracy, "precision":precision, "recall":recall}
            for i, metric in enumerate(raw_metrics.values()):
                plt.bar(x_axis + i*0.3, metric, width=0.3)
            plt.title('Metrics comparison')
            plt.ylabel('Metric')
            plt.xticks(x_axis, [str(x+1) for x in x_axis])
            plt.legend(list(raw_metrics.keys()))
            plt.xlabel('Cross validation number')
            plt.savefig(os.path.join(plot_results, "results_crval", "metrics.png"),dpi=400)

        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall) 
        cf_matrix = np.mean(np.array(cf_matrix), axis=0)

    return (accuracy, precision, recall), cf_matrix

def plot_data(data, path):
    if not os.path.exists(path):
        os.mkdir(path)
    fig, ax = plt.subplots()
    ax = sns.pairplot(data)
    ax.savefig(os.path.join(path, "metadata", "pairplot.png")) 

def print_metrics(metrics, metric_names=["Accuracy", "Precision", "Recall"]):
    # Metrics should be of the format acc, pre, recall
    for m, metric in enumerate(metrics):
        print("\t{:14s}: {:3.2f}".format(metric_names[m], metric))