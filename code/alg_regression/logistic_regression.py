"""
Logistic Regression Model
Rina Peshori
"""

import numpy as np
import os
import pandas

# Define the number of target classes as a constant
NUM_TARGET_CLASSES = 3

# Define base file path for data retrieval
BASE_PATH = "data/processed_data/"

# Define hyperparameters for model training
ITERATIONS = 5000 # define max. number of times we can iterate during alg. training
ETA = 0.01 # define standard learning rate (step size)
THRESHOLD = 1e-4 # define threshold for convergence

def sigmoid(z):
    # simple implementation of the sigmoid function
    z = np.clip(z, -500, 500) # constrain z to prevent overflow
    return 1 / (1 + np.exp(-z))

def get_train_data():
    # fetch training dataset from csv
    X_path = BASE_PATH + "X_train_scaled.csv"
    y_path = BASE_PATH + "y_train.csv"
    return _get_data_from_csv(X_path, y_path)

def get_test_data():
    # fetch test dataset from csv
    X_path = BASE_PATH + "X_test_scaled.csv"
    y_path = BASE_PATH + "y_test.csv"
    return _get_data_from_csv(X_path, y_path)
    
def _get_data_from_csv(X_path, y_path):
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

def init_matrices(X, y):
    num_rows = len(X)
    num_cols = len(X.columns)

    # convert dataframe to matrix (using numpy array)
    # add a column of ones so we can find our intercept
    X_matrix = np.c_[np.ones(num_rows), X.values]
    y_matrix = y.values

    # init beta matrix to zeros
    Beta_matrix = np.zeros(num_cols + 1) # +1 col of intercepts

    # ensure y is a 1-D vector for arithmetic
    y_matrix = y_matrix.reshape(-1) # -1 automatically reshapes into 1-D

    return X_matrix, y_matrix, Beta_matrix

def compute_loss(y, p):
    # compute value of loss function
    # in this case, cross-entropy
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1-epsilon)
    result = np.mean(y * np.log(p) + (1-y) * np.log(1-p)) # I love numpy
    # result = (-1/len(y)) * result
    return result

def train_oneVrest(X, y, num_classes):
    models = []

    for k in range(num_classes):
        y_binary = (y==k).astype(int)
        beta = train_bgd(X, y_binary)
        models.append(beta)

    return models

def train_bgd(X, y): # Batch Gradient Descent
    # get prediction results in typical linear regression style

    # create numpy matrices from given X and y
    X_matrix, y_matrix, Beta_matrix = init_matrices(X, y)

    # track the previously found loss value to measure convergence
    prev_loss = float('inf') # initialize to infinity

    for i in range(ITERATIONS):
        # compute probability predictions (X * Beta)
        predictions = np.dot(X_matrix, Beta_matrix)

        # apply sigmoid function to predicted probabilities
        p = sigmoid(predictions)

        # calculate the loss (cross-entropy)
        loss = compute_loss(y_matrix, p)
        # print(loss) # print out results if you want to check model stability

        # measure convergence using change in loss between iterations
        if abs(prev_loss - loss) < THRESHOLD:
            print(f"\tConverged at iteration {i}")
            # if convergence reached, exit loop
            break

        # if we're here, we haven't yet hit convergence
        # set this loss to our previous value
        prev_loss = loss

        # implement class weighting to stop model from ignoring lower-frequency classes
        num_pos = np.sum(y_matrix == 1)
        num_neg = np.sum(y_matrix == 0)
        pos_weight = len(y_matrix) / (2 * num_pos)
        neg_weight = len(y_matrix) / (2 * num_neg)
        weights = np.where(y_matrix==1, pos_weight, neg_weight)

        # compute gradient of cross-entropy loss
        transposed_X = X_matrix.T
        weighted = weights * (p - y_matrix)
        gradient = np.dot(transposed_X, weighted)

        # UPDATE beta accordingly!
        # using our assigned learning rate
        Beta_matrix = Beta_matrix - ETA * gradient

        if i == ITERATIONS-1:
            print("\tReached max. iteration number")

    return Beta_matrix

def make_decisions(X, model_Betas):
    # decide on classifications based on the output of our linear regression function
    
    # convert dataframe to matrix (using numpy array)
    # add a column of ones so we can find our intercept
    X_matrix = np.c_[np.ones(len(X)), X.values]

    probabilities = []
    for Beta in model_Betas:
        p = sigmoid(np.dot(X_matrix, Beta))
        probabilities.append(p)
    
    probabilities = np.array(probabilities)
    return np.argmax(probabilities, axis=0)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, num_classes):
    # find the true/false negatives & positives
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, predicted in zip(y_true, y_pred): # loop over true and predicted values simultaneously
        cm[true][predicted] += 1 # increment count for appropriate value based on true vs. predicted value
    return cm

# Run the custom-implemented Logistic Regression algorithm
# Supports custom test_X and y input
def run_algorithm_custom_test(test_X=pandas.DataFrame(), test_y=pandas.DataFrame()):
    custom_test_X = not test_X.empty # if given X is empty, no custom input was provided

    print("--- LOADING DATASETS ---")
    print("Loading training set...")
    train_X, train_y = get_train_data()
    if not custom_test_X:
        print("Loading test set...")
        test_X, test_y = get_test_data()
    else:
        print("Custom X and y sets provided. Proceeding with custom test set...")
    print("--- LOADING DATASETS COMPLETE ---")
    print()

    print("--- BEGINNING MODEL TRAINING ---")
    print("Training with One vs. Rest, Batch Gradient Descent (BGD)...")
    model_Betas = train_oneVrest(train_X, train_y, NUM_TARGET_CLASSES)
    print("--- MODEL TRAINING COMPLETE ---")
    print()

    print("--- BEGINNING PREDICTIONS ---")
    print("Predicting with One vs. Rest...")
    y_pred = make_decisions(test_X, model_Betas)
    print("--- PREDICTIONS COMPLETE ---")
    print()

    print("--- BEGINNING EVALUATION ---")
    # ensure test_y is a 1-D vector for arithmetic
    test_y_matrix = test_y.values.reshape(-1) # -1 automatically reshapes into 1-D
    print("Computing accuracy...")
    accuracy = compute_accuracy(test_y_matrix, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Constructing the confusion matrix...")
    cm = compute_confusion_matrix(test_y_matrix, y_pred, NUM_TARGET_CLASSES)
    print(cm)
    print("--- EVALUATION COMPLETE ---")
    print()

# Run the custom-implemented Logistic Regression algorithm
# Supports custom train_X and y input (will not perform evaluation on custom inputs)
def run_algorithm_custom_train(train_X=pandas.DataFrame(), train_y=pandas.DataFrame()):
    custom_train_X = not train_X.empty # if given X is empty, no custom input was provided

    print("--- LOADING DATASETS ---")
    if not custom_train_X:
        print("Loading training set...")
        train_X, train_y = get_train_data()
    else:
        print("Custom X and y sets provided. Proceeding with custom train set...")
    print("Loading test set...")
    test_X, test_y = get_test_data()
    print("--- LOADING DATASETS COMPLETE ---")
    print()

    print("--- BEGINNING MODEL TRAINING ---")
    print("Training with One vs. Rest, Batch Gradient Descent (BGD)...")
    model_Betas = train_oneVrest(train_X, train_y, NUM_TARGET_CLASSES)
    print("--- MODEL TRAINING COMPLETE ---")
    print()

    print("--- BEGINNING PREDICTIONS ---")
    print("Predicting with One vs. Rest...")
    y_pred = make_decisions(test_X, model_Betas)
    print("--- PREDICTIONS COMPLETE ---")
    print()

    print("--- BEGINNING EVALUATION ---")
    # ensure test_y is a 1-D vector for arithmetic
    test_y_matrix = test_y.values.reshape(-1) # -1 automatically reshapes into 1-D
    print("Computing accuracy...")
    accuracy = compute_accuracy(test_y_matrix, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Constructing the confusion matrix...")
    cm = compute_confusion_matrix(test_y_matrix, y_pred, NUM_TARGET_CLASSES)
    print(cm)
    print("--- EVALUATION COMPLETE ---")
    print()

def training_pipeline(train_X, train_y):
    # returns the classification function
    # NOTE: also returns the calculated model modifiers, which should be passed as a param to the classification function
    model_Betas = train_oneVrest(train_X, train_y, NUM_TARGET_CLASSES)
    return make_decisions, model_Betas

def main():
    run_algorithm_custom_test()
    print("Exiting... Thank you!")

if __name__ == "__main__":
    main()