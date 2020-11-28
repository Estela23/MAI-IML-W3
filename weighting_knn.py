# File where the three optional weighting functions will be implemented
import numpy as np
"""
return "feature_weights", after computing matrix_deltas and feature_weights we simply
multiply them and have the final distances between all the training and test elements
"""


def equal_weight(test_data):
    feature_weights = np.ones(test_data.shape[1])
    return feature_weights
