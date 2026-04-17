"""
Main pipeline file to run all models and provide results.
"""
import argparse
from analysis.data_splits import train_test, kfold_crossval
import random_forest.random_forest as rf

def run_kfold():
    """
    Run k-fold cross-validation on each of the models and output the results.
    This may take a while.
    """

    # Obtain an UNSCALED train/test split for use in k-fold cross-validation
    X_train, X_test, y_train, y_test = train_test()

    """
    Random Forest
    """
    print("-------RANDOM FOREST---------")
    tacc, vacc = kfold_crossval(X_train, y_train, rf.train)
    print(f"Training accuracy: {tacc}")
    print(f"Validation accuracy: {vacc}")
    

def run_train_test():
    print("Not implemented yet")

def main(kfold: bool):
    if kfold:
        run_kfold()
    else:
        run_train_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kfold",
        action="store_const",
        const=True,
        default=False,
        required=False,
        help="Whether or not to run k-fold validation",
    )
    args = parser.parse_args()
    main(args.kfold)