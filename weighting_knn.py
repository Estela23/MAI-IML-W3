# File where the three optional weighting functions will be implemented
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF

"""
return "feature_weights", after computing matrix_deltas and feature_weights we simply
multiply them and have the final distances between all the training and test elements
"""


def equal_weight(train_data):
    feature_weights = np.ones(train_data.shape[1])
    return feature_weights


def info_gain(train_data, train_classes, k):
    feature_weights = mutual_info_classif(train_data, train_classes, n_neighbors=k, copy=True, random_state=0)
    return feature_weights


def reliefF(train_data, train_classes, k):
    fs = ReliefF(n_neighbors=k, n_features_to_keep=train_data.shape[1])
    fs.fit(train_data, train_classes)
    feature_weights = fs.feature_scores
    return feature_weights
