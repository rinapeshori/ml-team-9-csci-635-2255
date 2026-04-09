import pandas as pd
import numpy as np
from TreeNode import TreeNode, DecideLow, DecideMedium, DecideHigh

def load_train_data():
    x_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')
    features = x_train.columns
    return x_train, y_train, features

def bootstrap_sample(x, y):
    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return x.iloc[indices], y.iloc[indices]

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
        
def construct_node(data: pd.DataFrame, target: np.ndarray, features, parent_level: int):
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
    if (parent_level == 5 or len(data) < 23):
        if len(data) == 0:
            return DecideLow(parent_level+1)
        return decisions[np.argmax(np.bincount(target))]
    for i in range(3):
        if len(target[target == i]) / len(target) > 0.85:
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
    this_node.under = construct_node(data[data[best_att] <= best_thresh], target[data[best_att] <= best_thresh], features, parent_level+1)
    this_node.over = construct_node(data[data[best_att] > best_thresh], target[data[best_att] > best_thresh], features, parent_level+1)
    return this_node

def classify(forest: list[TreeNode], x: pd.DataFrame):
    y = []
    for sample in x.iterrows():
        bins = [0, 0, 0]
        for tree in forest:
            bins[tree.decide(sample[1])] += 1
        y.append(np.argmax(bins))
        
    return np.array(y)

def main():
    # Load the training data
    x_train, y_train, features = load_train_data()

    # Make the random forest
    trees = []
    for _ in range(20):
        xb, yb = bootstrap_sample(x_train, y_train)
        trees.append(construct_node(xb, yb["burnout_risk"], features, 0))

    # Run!
    y_train_pred = classify(trees, x_train)
    print(len(y_train_pred[y_train_pred == y_train["burnout_risk"]]) / len(y_train))

if __name__ == "__main__":
    main()