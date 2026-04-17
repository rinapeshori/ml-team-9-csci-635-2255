import pandas as pd
import numpy as np
from TreeNode import TreeNode, DecideLow, DecideMedium, DecideHigh
from sklearn.preprocessing import StandardScaler

# Hyperparameters
NUM_TREES = 10 # Number of trees in the forest

# Early stopping criteria
MAX_DEPTH = 7 # Max depth of the decision trees
MIN_SPLITTING_SIZE = 15 # Sample count below which the tree should decide for the plurality class
DECISION_THRESHOLD = 0.85 # If a tree node has above this percentage of samples in one class, the tree decides for that class

def load_data(type: str):
    x_train = pd.read_csv(f'data/processed_data/X_{type}_scaled.csv')
    y_train = pd.read_csv(f'data/processed_data/y_{type}.csv')
    features = x_train.columns
    return x_train, y_train, features

def bootstrap_sample(x, y):
    """
    Construct a bootstrap sample with replacement from the x/y dataset
    """
    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return x.iloc[indices].reset_index(), y.iloc[indices].reset_index()["burnout_risk"]

def log2(x: float):
    """
    Takes the log base 2 of x, returning 0 if x is 0 to avoid undefined behavior.
    """
    if x == 0:
        return 0
    return np.log2(x)

def entropy(classes: np.ndarray):
    """
    Calculates the entropy of a set of class values. Assumes that the class values are either 0, 1, or 2.
    """
    if(len(classes) == 0):
        return 0
    prob_0 = len(classes[classes == 0]) / len(classes)
    prob_1 = len(classes[classes == 1]) / len(classes)
    prob_2 = len(classes[classes == 2]) / len(classes)
    return - (prob_1*log2(prob_1) + prob_2*log2(prob_2) + prob_0*log2(prob_0))
    
def best_avg_entropy(data: pd.DataFrame, target: np.ndarray, feature_name: str):
    """
    Finds the minimum average entropy in the data when splitting on the given feature and returns that value and the threshold at which it occurs.
    """
    feature = np.array(data[feature_name])
    avg_entropies = []
    thresh_range = np.linspace(np.min(feature), np.max(feature), num=50)
    for thresh in thresh_range:
        under_classes = target[feature <= thresh]
        over_classes = target[feature > thresh]
        avg_entropies.append(entropy(under_classes)*(len(under_classes)/len(target)) + entropy(over_classes)*(len(over_classes)/len(target)))
    np_entropies = np.array(avg_entropies)
    return np.min(np_entropies), thresh_range[np.argmin(np_entropies)]
        
def construct_node(data: pd.DataFrame, target: np.ndarray, features, parent_level: int, max_depth: int = MAX_DEPTH, min_size: int = MIN_SPLITTING_SIZE, decision_thresh: float = DECISION_THRESHOLD):
    """
    Recursively construct a decision tree using a custom node structure.
    """

    # Check for a base case: if this is the fifth level of the structure, the 
    # training data has fewer than 23 samples in this node, or more than 85% of
    # the data belongs to one class, return a node representing the decision of 
    # either -1 or 1
    decisions = {
        0: DecideLow(parent_level+1),
        1: DecideMedium(parent_level+1),
        2: DecideHigh(parent_level+1)
    }
    if (parent_level == max_depth or len(data) < min_size):
        if len(data) == 0:
            return DecideLow(parent_level+1)
        return decisions[np.argmax(np.bincount(target))]
    for i in range(3):
        if len(target[target == i]) / len(target) > decision_thresh:
            return decisions[i] 
        
    # Decide which attribute is the best to split on and at which threshold
    best_entropy = 1
    best_thresh = 0
    best_att = ""
    for att in features:
        ent, thresh = best_avg_entropy(data, target, att)
        if ent < best_entropy:
            best_entropy = ent
            best_thresh = thresh
            best_att = att

    # Construct a new decision node for this level. Recursively call this 
    # function to construct more nodes for splitting the data on each side of 
    # the current threshold.
    this_node = TreeNode(best_att, best_thresh, parent_level+1)
    under = data[best_att] <= best_thresh
    over = data[best_att] > best_thresh
    this_node.under = construct_node(data[under], target[under], features, parent_level+1, max_depth, min_size, decision_thresh)
    this_node.over = construct_node(data[over] > best_thresh, target[over], features, parent_level+1, max_depth, min_size, decision_thresh)
    return this_node

def classify(forest: list[TreeNode], x: pd.DataFrame):
    """
    Use the random forest to make a plurality/majority decision on each input sample.
    """
    y = []
    for sample in x.iterrows():
        bins = [0, 0, 0]
        for tree in forest:
            bins[tree.decide(sample[1])] += 1
        y.append(np.argmax(bins))
        
    return np.array(y)

def test_for_fine_tuning():
    """
    Test a set of hyperparameters to find which ones find the ideal bias-variance tradeoff between validation and training data
    """

def train(x: pd.DataFrame, y: pd.DataFrame):
    """
    Trains a decision tree model on the x and y training sets and returns a method to classify future sets of samples
    """
    features = x.columns
    trees = []
    for _ in range(NUM_TREES):
        xb, yb = bootstrap_sample(x, y)
        trees.append(construct_node(xb, yb, features, 0))

    return lambda x: classify(trees, x)

def main():
    # Load the training data
    # x_train, x_test, y_train, y_test = train_test()
    # features = x_train.columns
    
    # # Run k-fold cross validation on the model
    # t_acc, v_acc = kfold_crossval(x_train, y_train, train)
    # print(f"Training accuracy: {t_acc}")
    # print(f"Validation accuracy: {v_acc}")

    # # Make the random forest on all training data
    # sc = StandardScaler()
    # x_train = pd.DataFrame(sc.fit_transform(x_train), columns=x_train.columns)
    # x_test = pd.DataFrame(sc.transform(x_test), columns=x_test.columns)
    # trees = []
    # for _ in range(NUM_TREES):
    #     xb, yb = bootstrap_sample(x_train, y_train)
    #     trees.append(construct_node(xb, yb, features, 0))

    # y_train_pred = classify(trees, x_train)
    # print(f"Training accuracy: {len(y_train_pred[y_train_pred == y_train]) / len(y_train)}")

    # # Run on test data
    # y_test_pred = classify(trees, x_test)
    # print(f"Test accuracy: {len(y_test_pred[y_test_pred == y_test]) / len(y_test)}")
    pass

if __name__ == "__main__":
    main()