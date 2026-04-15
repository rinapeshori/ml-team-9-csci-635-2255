"""
Synthetic Data Generation
Rina Peshori

IMPORTANT -- RUN WITH COMMAND SIMILAR TO BELOW
'py -m code.augmentation.main'
"""
from ..alg_regression import logistic_regression

import numpy as np
import os
import pandas

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11

# Define base file path for data retrieval
BASE_PATH = "data/processed_data/"

def get_train_data():
    # fetch training dataset from csv
    X_path = BASE_PATH + "X_train_scaled.csv"
    y_path = BASE_PATH + "y_train.csv"
    return _get_data_from_csv(X_path, y_path)
    
def _get_data_from_csv(X_path, y_path):
    # fetch a dataset from given csv filepaths
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        # TODO: Ensure that program execution halts if error is encountered here.
        print(f"Error: File not found.")
        return None, None
    try :
        X = pandas.read_csv(X_path, low_memory=False)
        y = pandas.read_csv(y_path, low_memory=False)
        return X, y
    except pandas.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except pandas.errors.ParserError as e:
        print(f"Error: Failed to parse CSV - {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    return None, None

def main():
    n_samples = 2000 # number of samples to generate

    X, y = get_train_data()
    kdes = {} # fit a different kde model for each class

    # ensure y is 1-D
    y_arr = y.values.reshape(-1)
    classes, counts = np.unique(y_arr, return_counts=True)

    # Fit a kernel density model using GridSearchCV to determine the best parameter for bandwidth
    bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
    for cls in classes:
        X_cls = X[y_arr == cls]
        grid_search = GridSearchCV(KernelDensity(), bandwidth_params, cv=5)
        grid_search.fit(X_cls.values)
        kde = grid_search.best_estimator_
        kdes[cls] = kde

    # Maintain approximately original class distribution
    freqs = counts / counts.sum()
    samples_per_class = np.floor(freqs * n_samples).astype(int)
    remainder = n_samples - samples_per_class.sum() # account for weird division/flooring cutting things off
    if remainder > 0:
        idxs = np.argsort(-counts)
        for i in range(remainder):
            samples_per_class[idxs[i % len(classes)]] += 1

    # Generate/sample some new data from this dataset
    new_X = []
    new_y = []
    for cls, n in zip(classes, samples_per_class):
        new_data = kdes[cls].sample(n, random_state=rand_state)
        new_X.append(new_data)
        new_y.append(np.full(n, cls))

    # Format the data
    X_synthset = np.vstack(new_X)
    y_synthset = np.concatenate(new_y)
    df_X_synth = pandas.DataFrame(X_synthset, columns=X.columns)
    df_y_synth = pandas.Series(y_synthset)

    print(df_X_synth)
    print()
    print()
    print(df_y_synth)

    logistic_regression.run_algorithm(df_X_synth, df_y_synth)

if __name__ == "__main__":
    main()