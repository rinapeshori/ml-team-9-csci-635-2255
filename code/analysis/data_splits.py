import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FILE = "data/wfh_burnout_dataset.csv"

def train_test(split: float = 0.2):
    """
    Obtain a train/test split from our dataset.
    """
    data = pd.read_csv(DATA_FILE)
    X = data.drop(["burnout_risk", "burnout_score"], axis=1)
    y = data["burnout_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def kfold_crossval(X_train, y_train, fitfunc, k=10):
    """
    Perform k-fold cross validation with the given fitting function for a model
    on the training data.
    """
    permutation = np.random.Generator.permutation(range(len(X_train)))
    X = X_train[permutation]
    y = y_train[permutation]
    t_acc = 0
    v_acc = 0
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)

    for i in range(k):
        # Compute folds for this cycle
        X_val = X_folds[i]
        y_val = y_folds[i]
        indices = [j for j in range(k)].pop(i)
        X_train = np.concat(X_folds[indices])
        y_train = np.concat(y_folds[indices])

        # Make classifications and get training and validation accuracy
        y_train_pred = fitfunc(X_train)
        y_val_pred = fitfunc(X_val)
        t_acc += len(y_train_pred[y_train_pred == y_train["burnout_risk"]]) / len(y_train)
        v_acc += len(y_val_pred[y_val_pred == y_val["burnout_risk"]]) / len(y_val)

    return t_acc/k, v_acc/k