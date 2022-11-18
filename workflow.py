
def import_data(csv_path, col_names, normalize="std", dummy_cols=None, replace_nan="mean"):
    """
    csv_path: path to the data in csv format
    col_names: names of the columns of the dataset
    normalize: Normalization type for all columns. ("std", "0_1")\
    dummy_cols: names of the columns to apply dummy variables
    replace_nan: Way to replace nan or missing values. ("mean", "median", "remove")
    """
    return data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
import seaborn as sns
import os 


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

    if cross_validation is None:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        cf_matrix = confusion_matrix(y_pred, y_test)

        if plot_results is not None:    
            if os.path.exists(os.path.join(plot_results, "results")):
                os.mkdir(os.path.join(plot_results, "results"))
            ax = sns.heatmap(cf_matrix, annot=True)
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

        for train_index, test_index in cross_val.split(X):
            # Copy the model so it can be trained from 0 each time
            model_ = clone(model)
            X_train, X_test, y_train, y_test = data[train_index], data[test_index], labels[train_index], labels[test_index]
            # Fit and predict the model
            model_.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Save the parameters of the cross validation
            accuracy.append(accuracy_score(y_pred, y_test))
            precision.append(precision_score(y_pred, y_test))
            recall.append(recall_score(y_pred, y_test))
            cf_matrix.append(confusion_matrix(y_pred, y_test))

        if plot_results is not None:    
            if os.path.exists(os.path.join(plot_results, "results_crval")):
                os.mkdir(os.path.join(plot_results, "results_crval"))
            # Plot the mean confusion matrix
            ax = sns.heatmap(np.mean(np.array(cf_matrix), axis=0), annot=True)
            ax.set_title('Mean Confusion Matrix')
            ax.set_ylabel('True values')
            ax.set_xlabel('Predicted values')
            fig = ax.get_figure()
            fig.savefig(os.path.join(plot_results, "results_crval", "confusion_matrix.png"))

            # Plot the metric comparison
            raw_data = {"accuracy":accuracy, "precision":precision, "recall":recall}
            ax = sns.barplot(data=raw_data) # TODO 
            ax.set_title('Metrics comparison')
            ax.set_ylabel('Metric')
            ax.set_xlabel('Cross validation')
            ax.set_legend([])
            fig = ax.get_figure()
            fig.savefig(os.path.join(plot_results, "results_crval", "metrics.png"))

    return (accuracy, precision, recall), cf_matrix

def plot_data(data, path):
    
    ax = sns.pairplot(data)
    fig = ax.get_figure()
    fig.savefig(os.path.join(path, "metadata", "pairplot.png")) 
