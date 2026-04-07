import pandas as pd
import numpy as np
from TreeNode import TreeNode, DecideLow, DecideMedium, DecideHigh

def load_train_data():
    x_train = pd.read_csv('data/X_train_scaled.csv')
    y_train = pd.read_csv('data/y_train.csv')
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
    
def best_avg_entropy(data: pd.DataFrame, target: np.ndarray,feature_name: str):
    """
    Finds the minimum average entropy in the data when splitting on the given feature and returns that value and the threshold at which it occurs.
    """
    feature = np.array(data[feature_name])
    avg_entropies = []
    for thresh in range(np.min(feature), np.max(feature)+1):
        under_classes = target[feature <= thresh]
        over_classes = target[feature > thresh]
        avg_entropies.append(entropy(under_classes)*(len(under_classes)/len(target)) + entropy(over_classes)*(len(over_classes)/len(target)))
    np_entropies = np.array(avg_entropies)
    return np.min(np_entropies), np.argmin(np_entropies)+np.min(feature)
        
def construct_node(data: pd.DataFrame, target: np.ndarray,parent_level: int):
    """
    Recursively construct a decision tree using a custom node structure.
    """

    # Check for a base case: if this is the fifth level of the structure, the 
    # training data has fewer than 23 samples in this node, or more than 85% of
    # the data belongs to one class, return a node representing the decision of 
    # either -1 or 1
    pct_0 = len(target[target == 0]) / len(target)
    pct_1 = len(target[target == 1]) / len(target)
    pct_2 = len(target[target == 2]) / len(target)
    if (pct_0 > 0.85):
        return DecideLow(parent_level+1)
    elif (pct_1 > 0.85):
        return DecideMedium(parent_level+1)
    elif (pct_2 > 0.85):        
        return DecideHigh(parent_level+1)
    if (parent_level == 4 or len(data) < 23):
        
        
    # Decide which attribute is the best to split on and at which threshold
    best_entropy = 1
    best_thresh = 0
    best_att = ""
    for att in ATTRIBUTES:
        ent, thresh = best_avg_entropy(data, att)
        if ent < best_entropy:
            best_entropy = ent
            best_thresh = thresh
            best_att = att

    # Construct a new decision node for this level. Recursively call this 
    # function to construct more nodes for splitting the data on each side of 
    # the current threshold.
    this_node = TreeNode(best_att, best_thresh, parent_level+1)
    this_node.under = construct_node(data[data[best_att] <= best_thresh], parent_level+1)
    this_node.over = construct_node(data[data[best_att] > best_thresh], parent_level+1)
    return this_node

def build_tree(x, y, max_depth=None):
    

def main():
    # Load the training data
    x_train, y_train, features = load_train_data()


if __name__ == "__main__":
    main()