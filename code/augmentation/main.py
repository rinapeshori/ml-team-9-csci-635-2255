"""
Synthetic Data Generation
Rina Peshori

IMPORTANT -- RUN WITH COMMAND SIMILAR TO BELOW
'py -m code.augmentation.main'
"""
# from ..alg_regression import logistic_regression

import numpy as np
import os
import pandas

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11

# Define base file path for data retrieval
BASE_PATH = "data/"
SCALED_BASE_PATH = "data/processed_data/"

def get_train_data():
    # fetch UNSCALED training dataset from csv
    path = BASE_PATH + "wfh_burnout_dataset.csv"
    if os.path.exists(path):
        try:
            ds = pandas.read_csv(path, low_memory=False)
            # drop unnecessary feature
            ds = ds.drop("user_id", axis=1)
            # drop unnecessary target
            ds = ds.drop("burnout_score", axis=1)
            # encode string column(s)
            ds = pandas.get_dummies(ds, columns=["day_type"], drop_first=True) # CHANGE:
            # ds = pandas.get_dummies(ds, columns=["day_type"], drop_first=True)
            # encode target variable
            ds["burnout_risk"] = ds["burnout_risk"].map({"High": 2, "Medium": 1, "Low": 0})
            # separate features from targets
            X = ds.drop("burnout_risk", axis=1)
            y = ds["burnout_risk"]
            return X, y
        except pandas.errors.EmptyDataError:
            print("Error: CSV file is empty.")
        except pandas.errors.ParserError as e:
            print(f"Error: Failed to parse CSV - {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
        return None, None
    else:
        print(f"Error: File not found.")
        return None, None

def get_scaled_train_data():
    # fetch scaled training dataset from csv
    X_path = SCALED_BASE_PATH + "X_train_scaled.csv"
    y_path = SCALED_BASE_PATH + "y_train.csv"
    # fetch a dataset from given csv filepaths
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        # Ensure that program execution halts if error is encountered here.
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

def _scale_data(X):
    train_X_raw, _ = get_train_data()
    scaler = StandardScaler().fit(train_X_raw)
    X_scaled = scaler.transform(X)
    X_scaled = pandas.DataFrame(X_scaled, columns=X.columns)
    return X_scaled

def _apply_constraints(df, constraints):
    """
    constraints: dict mapping column name -> spec
    spec examples:
        {"type": "categorical", "values": [0, 1]}
        {"type": "int", "min": 0, "max": 20}
    """
    if not constraints:
        return df
    df = df.copy()
    for col, spec in constraints.items():
        if col not in df.columns:
            continue
        # operate on numeric representation
        vals = df[col].astype(float).to_numpy()
        t = spec.get("type")
        if t == "categorical":
            allowed = np.array(spec.get("values", []))
            if allowed.size == 0:
                continue
            idx = np.abs(vals[:, None] - allowed[None, :]).argmin(axis=1)
            df[col] = allowed[idx]
        elif t == "int":
            mn = spec.get("min", None)
            mx = spec.get("max", None)
            ints = np.rint(vals).astype(int)
            if mn is not None or mx is not None:
                if mn is None:
                    mn = ints.min()
                if mx is None:
                    mx = ints.max()
                ints = np.clip(ints, mn, mx)
            df[col] = ints
        elif t == "float":
            mn = spec.get("min", None)
            mx = spec.get("max", None)
            fvals = vals
            if mn is not None or mx is not None:
                if mn is None:
                    mn = fvals.min()
                if mx is None:
                    mx = fvals.max()
                fvals = np.clip(fvals, mn, mx)
            df[col] = fvals
        else:
            mn = spec.get("min", None)
            mx = spec.get("max", None)
            if mn is not None or mx is not None:
                df[col] = np.clip(vals, mn if mn is not None else vals.min(), mx if mx is not None else vals.max())
    return df

def _data_gen(X, y, n_samples, constraints=None):
    # generates synthetic X and y datasets given original X and y and number of samples to generate
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

    # Apply constraints (if provided) to enforce categorical/int/float ranges
    df_X_synth = _apply_constraints(df_X_synth, constraints)

    return df_X_synth, df_y_synth

def generate_unscaled_synthetic_data(n_samples=2000):
    X, y = get_train_data()
    print(X)
    print(y)
    # provide constraints to the generated unscaled data (based on original dataset constraints)
    # float constraints shouldn't be necessary, but are included here just in case
    constraints = {
        "day_type_Weekend": {"type": "categorical", "values": [0, 1]}, # CHANGE:
        "work_hours": {"type": "float", "min": 0.5, "max": 18},
        "screen_time_hours": {"type": "float", "min": 0, "max": 18},
        "meetings_count": {"type": "int", "min": 0, "max": 20},
        "breaks_taken": {"type": "int", "min": 0, "max": 15},
        "after_hours_work": {"type": "categorical", "values": [0, 1]},
        "app_switches": {"type": "int", "min": 5, "max": 200},
        "sleep_hours": {"type": "float", "min": 3, "max": 10},
        "task_completion": {"type": "float", "min": 0, "max": 100},
        "isolation_index": {"type": "int", "min": 3, "max": 9},
        "fatigue_score": {"type": "float", "min": 0, "max": 10},
    }

    newX, newY = _data_gen(X, y, n_samples, constraints=constraints)

    print(newX)
    print()
    print()
    print(newY)

    return newX, newY

def augment_training_data(X, y, n_samples=2000):
    if isinstance(y, pandas.DataFrame):
        y = y.squeeze()
    else:
        y = y.squeeze() if hasattr(y, "squeeze") else y

    df_X_synth, df_y_synth = _data_gen(X, y, n_samples)

    X_aug = pandas.concat([X, df_X_synth], ignore_index=True)
    y_aug = pandas.concat([y, df_y_synth], ignore_index=True)

    return X_aug, y_aug

def augment_training_data_unscaled(X, y, n_samples=2000):
    if isinstance(y, pandas.DataFrame):
        y = y.squeeze()
    else:
        y = y.squeeze() if hasattr(y, "squeeze") else y

    constraints = {
        "day_type_Weekend": {"type": "categorical", "values": [0, 1]},
        "work_hours": {"type": "float", "min": 0.5, "max": 18},
        "screen_time_hours": {"type": "float", "min": 0, "max": 18},
        "meetings_count": {"type": "int", "min": 0, "max": 20},
        "breaks_taken": {"type": "int", "min": 0, "max": 15},
        "after_hours_work": {"type": "categorical", "values": [0, 1]},
        "app_switches": {"type": "int", "min": 5, "max": 200},
        "sleep_hours": {"type": "float", "min": 3, "max": 10},
        "task_completion": {"type": "float", "min": 0, "max": 100},
        "isolation_index": {"type": "int", "min": 3, "max": 9},
        "fatigue_score": {"type": "float", "min": 0, "max": 10},
    }

    df_X_synth, df_y_synth = _data_gen(X, y, n_samples, constraints=constraints)

    X_aug = pandas.concat([X, df_X_synth], ignore_index=True)
    y_aug = pandas.concat([y, df_y_synth], ignore_index=True)

    return X_aug, y_aug

def main():
    newX, newY = generate_unscaled_synthetic_data()
    newX_scaled = _scale_data(newX)
    # n_samples = 2000 # number of samples to generate

    # X, y = get_scaled_train_data()
    # newX, newY = _data_gen(X, y, n_samples)

    # print(newX)
    # print()
    # print()
    # print(newY)

    # logistic_regression.run_algorithm_custom_train(newX_scaled, newY)

    ## TODO:
    # - convert to unscaled data generation (or give it as an option)
    # - export as csv (can be separate from original dataset)
    # - make sure there's a function that JUST produces that unscaled csv file for pipeline integration

if __name__ == "__main__":
    main()
