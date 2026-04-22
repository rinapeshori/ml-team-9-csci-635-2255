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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from analysis.data_splits import train_test, kfold_crossval
from augmentation.main import augment_training_data, augment_training_data_unscaled

USE_UNSCALED = True # True - unscaled augmentation, False - scaled augmentation

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
    if augment:
        print(f"Data Augmentation Type: {'UNSCALED' if USE_UNSCALED else 'SCALED'}")
    print()

def run_kfold(use_augmentation: bool, model, X_train, y_train):
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()

    n = X_train.shape[0]
    k = 10
    rng = np.random.RandomState(42)
    permutation = rng.choice(n, size=n, replace=False)
    indices = np.array_split(permutation, k)

    t_acc = 0.0
    v_acc = 0.0

    if use_augmentation:
        print("\n-------K-FOLD (WITH AUGMENTATION)---------")
    else:
        print("\n-------K-FOLD (NO AUGMENTATION)---------")

    for i in range(k):
        print(f"\nFold {i}")

        X_val = X_train.iloc[indices[i]]
        y_val = y_train.iloc[indices[i]]

        X_fold_train = X_train.drop(X_train.index[indices[i]])
        y_fold_train = y_train.drop(y_train.index[indices[i]])

        if use_augmentation:
            if USE_UNSCALED:
                # UNCALED AUGMENTATION
                X_train_aug_raw, y_train_final = augment_training_data_unscaled(
                    X_fold_train,
                    y_fold_train,
                    n_samples=len(X_fold_train)
                )

                scaler = StandardScaler()
                X_train_final = pd.DataFrame(
                    scaler.fit_transform(X_train_aug_raw),
                    columns=X_train.columns
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_train.columns
                )

            else:
                # SCALED AUGMENTATION
                scaler = StandardScaler()
                X_fold_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_fold_train),
                    columns=X_train.columns
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_train.columns
                )

                X_train_final, y_train_final = augment_training_data(
                    X_fold_train_scaled,
                    y_fold_train,
                    n_samples=len(X_fold_train_scaled)
                )
        else:
            scaler = StandardScaler()
            X_train_final = pd.DataFrame(
                scaler.fit_transform(X_fold_train),
                columns=X_train.columns
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_train.columns
            )
            y_train_final = y_fold_train

        fitfunc = model.train(X_train_final, y_train_final)

        y_train_pred = fitfunc(X_train_final)
        y_val_pred = fitfunc(X_val_scaled)

        fold_train_acc = np.mean(y_train_pred == y_train_final.to_numpy())
        fold_val_acc = np.mean(y_val_pred == y_val.to_numpy())

        print(f"Training Acc: {fold_train_acc:.4f}")
        print(f"Validation Acc: {fold_val_acc:.4f}")

        t_acc += fold_train_acc
        v_acc += fold_val_acc

    print("\nFINAL AVERAGE ACCURACIES")
    print(f"Training accuracy: {t_acc / k:.4f}")
    print(f"Validation accuracy: {v_acc / k:.4f}")


def run_train_test(use_augmentation: bool, model, model_name, X_train, X_test, y_train, y_test):
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    if use_augmentation:
        print("\n-------TRAIN/TEST (WITH AUGMENTATION)---------")

        if USE_UNSCALED:
            # UNSCALED AUGMENTATION
            X_train_aug_raw, y_train_final = augment_training_data_unscaled(
                X_train,
                y_train,
                n_samples=len(X_train)
            )

            scaler = StandardScaler()
            X_train_final = pd.DataFrame(
                scaler.fit_transform(X_train_aug_raw),
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_train.columns
            )

        else:
            # SCALED AUGMENTATION
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_train.columns
            )

            X_train_final, y_train_final = augment_training_data(
                X_train_scaled,
                y_train,
                n_samples=len(X_train_scaled)
            )

    else:
        print("\n-------TRAIN/TEST (NO AUGMENTATION)---------")

        scaler = StandardScaler()
        X_train_final = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_train.columns
        )
        y_train_final = y_train

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

    cm_array = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig(f"cm_{model_name}.png")
    plt.show()
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    print("\nPer-Class Accuracy:")
    y_test_np = y_test.to_numpy()
    for cls in np.unique(y_test_np):
        mask = (y_test_np == cls)
        cls_acc = np.mean(y_test_pred[mask] == y_test_np[mask])
        print(f"Class {cls} Accuracy: {cls_acc:.4f}")


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
            run_train_test(args.augment, model, name, X_train, X_test, y_train, y_test)

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
