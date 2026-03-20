import pandas as pd
import numpy as np
from sklearn import svm

DATA_DIR = "../data/processed_data"

def svm_decide(model: svm.SVC, data: pd.DataFrame):
    return model.predict(data)

def print_comparison(predictions, actual):
    correct = np.sum(predictions == actual)
    total = len(actual)
    accuracy = correct / total
    print(f"Correct predictions: {correct}")
    print(f"Number of samples: {total}")
    print(f"Accuracy: {accuracy:.2%}")

def main():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_scaled.csv")
    y_train = np.array(pd.read_csv(f"{DATA_DIR}/y_train.csv")).reshape(-1)

    model = svm.SVC(kernel='linear')
    
    model.fit(X_train, y_train)
    
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_scaled.csv")
    predictions = svm_decide(model, X_test)
    actual = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.reshape(-1)
    print_comparison(predictions, actual)


if __name__ == "__main__":
    main()