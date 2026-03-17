import pandas as pd
import numpy as np
from sklearn import svm

DATA_DIR = "../data/processed_data"

def main():
    # Load the training data
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_scaled.csv")
    y_train = np.array(pd.read_csv(f"{DATA_DIR}/y_train.csv")).reshape(-1)

    # Initialize the SVM model
    model = svm.SVC(kernel='linear')
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    print("SVM model trained successfully.")


if __name__ == "__main__":
    main()