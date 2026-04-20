"""
Main pipeline file to run models and provide results.
IMPORTANT -- RUN WITH COMMAND SIMILAR TO:
    python main.py --mlp
    python main.py --logistic
    python main.py --random_forest
    python main.py --all

OPTIONS:
    --kfold     Run k-fold validation
    --augment   Use data augmentation

EXAMPLES:
    python main.py --mlp --augment
    python main.py --all --kfold --augment
"""
import argparse
import importlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from analysis.data_splits import train_test, kfold_crossval
from augmentation.main import augment_training_data

MODEL_MAP = {
    "mlp": "MLP.mlp",
    "logistic": "alg_regression.logistic_regression",
    "random_forest": "random_forest.random_forest",
}

def load_model(model_name: str):
    return importlib.import_module(MODEL_MAP[model_name])

def print_config(kfold: bool, augment: bool, model_name: str):
    print()
    print("MODEL CONFIGURATION")
    print(f"Model: {model_name.upper()}")
    print(f"Mode: {'K-FOLD' if kfold else 'TRAIN/TEST'}")
    print(f"Data Augmentation: {'ON' if augment else 'OFF'}")
    print()

def run_kfold(use_augmentation: bool, model, X_train, y_train):
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    if not use_augmentation:
        print("\n-------K-FOLD (NO AUGMENTATION)---------")
        tacc, vacc = kfold_crossval(X_train, y_train, model.train)
        print(f"Training accuracy: {tacc:.4f}")
        print(f"Validation accuracy: {vacc:.4f}")
        return
    n = X_train.shape[0]
    k = 10
    rng = np.random.RandomState(42)
    permutation = rng.choice(n, size=n, replace=False)
    indices = np.array_split(permutation, k)
    t_acc = 0.0
    v_acc = 0.0
    print("\n-------K-FOLD (WITH AUGMENTATION)---------")
    for i in range(k):
        print(f"\nFold {i}")

        X_val = X_train.iloc[indices[i]]
        y_val = y_train.iloc[indices[i]]

        X_fold_train = X_train.drop(X_train.index[indices[i]])
        y_fold_train = y_train.drop(y_train.index[indices[i]])

        scaler = StandardScaler()
        X_fold_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_fold_train),
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_train.columns
        )
        X_train_aug, y_train_aug = augment_training_data(
            X_fold_train_scaled,
            y_fold_train,
            n_samples=len(X_fold_train_scaled)
        )
        fitfunc = model.train(X_train_aug, y_train_aug)
        y_train_pred = fitfunc(X_train_aug)
        y_val_pred = fitfunc(X_val_scaled)

        fold_train_acc = np.mean(y_train_pred == y_train_aug.to_numpy())
        fold_val_acc = np.mean(y_val_pred == y_val.to_numpy())

        print(f"Training Acc: {fold_train_acc:.4f}")
        print(f"Validation Acc: {fold_val_acc:.4f}")

        t_acc += fold_train_acc
        v_acc += fold_val_acc

    print("\nFINAL AVERAGE ACCURACIES")
    print(f"Training accuracy: {t_acc / k:.4f}")
    print(f"Validation accuracy: {v_acc / k:.4f}")


def run_train_test(use_augmentation: bool, model, X_train, X_test, y_train, y_test):
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns
    )
    if use_augmentation:
        print("\n-------TRAIN/TEST (WITH AUGMENTATION)---------")
        X_train_final, y_train_final = augment_training_data(
            X_train_scaled,
            y_train,
            n_samples=len(X_train_scaled)
        )
    else:
        print("\n-------TRAIN/TEST (NO AUGMENTATION)---------")
        X_train_final, y_train_final = X_train_scaled, y_train

    fitfunc = model.train(X_train_final, y_train_final)
    y_train_pred = fitfunc(X_train_final)
    y_test_pred = fitfunc(X_test_scaled)
    train_acc = np.mean(y_train_pred == y_train_final.to_numpy())
    test_acc = np.mean(y_test_pred == y_test.to_numpy())

    print("\nRESULTS")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    cm = pd.crosstab(
        pd.Series(y_test.to_numpy(), name="Actual"),
        pd.Series(y_test_pred, name="Predicted")
    )
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))


def main(args):
    if args.all:
        models = ["mlp", "logistic", "random_forest"]
    else:
        models = []
        if args.mlp:
            models.append("mlp")
        if args.logistic:
            models.append("logistic")
        if args.random_forest:
            models.append("random_forest")
    if not models:
        print("Please select at least one model flag: --mlp / --logistic / --random_forest / --all")
        return
    X_train, X_test, y_train, y_test = train_test()
    for name in models:
        model = load_model(name)
        print_config(args.kfold, args.augment, name)
        if args.kfold:
            run_kfold(args.augment, model, X_train, y_train)
        else:
            run_train_test(args.augment, model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--logistic", action="store_true")
    parser.add_argument("--random_forest", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kfold", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    main(args)