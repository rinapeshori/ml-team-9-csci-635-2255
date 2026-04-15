import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = "data/wfh_burnout_dataset.csv"

def train_test(split = 0.2):
    """
    Obtain a train/test split from our dataset.
    """
    data = pd.read_csv(DATA_FILE)
    data = pd.get_dummies(data, columns=["day_type"], drop_first=True)
    data["burnout_risk"] = data["burnout_risk"].map({"High": 2, "Medium": 1, "Low": 0})
    X = data.drop(["burnout_risk", "burnout_score", "user_id"], axis=1)
    y = data["burnout_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split,
        stratify=y,
        random_state=42
    )
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train, X_test, y_train, y_test

def kfold_crossval(X: pd.DataFrame, y: pd.DataFrame, tfunc, k=10):
    """
    Perform k-fold cross validation with the given fitting function for a model
    on the training data.
    """
    n = X.shape[0]
    
    # X = X_train.iloc[permutation]
    # y = y_train.iloc[permutation]

    X_folds = []
    y_folds = []
    permutation = np.random.choice(n, size=n, replace=False)
    indices = np.array_split(permutation, k)

    t_acc = 0
    v_acc = 0
    for i in range(k):
        # Compute folds for this cycle
        print(f"Round {i}")
        X_val = X.iloc[indices[i]]
        y_val = y.iloc[indices[i]]
        # indices = [j for j in range(k)].pop(i)
        X_train = X.drop(X.index[indices[i]])
        y_train = y.drop(y.index[indices[i]])

        # Make classifications and get training and validation accuracy
        sc = StandardScaler()
        X_train_scaled = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
        X_val_scaled = pd.DataFrame(sc.transform(X_val), columns=X.columns)
        fitfunc = tfunc(X_train_scaled, y_train)   
        y_train_pred = fitfunc(X_train_scaled)
        y_val_pred = fitfunc(X_val_scaled)
        t_acc += len(y_train_pred[y_train_pred == y_train]) / len(y_train)
        v_acc += len(y_val_pred[y_val_pred == y_val]) / len(y_val)

    return t_acc/k, v_acc/k