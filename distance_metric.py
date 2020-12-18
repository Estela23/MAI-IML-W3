# File where the three optional distances will be implemented
import numpy as np


def manhattan_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.sum(feature_weights * (np.abs(train_matrix[:, None, :] -
                                                        test_matrix[None, :, :])),
                              axis=-1)

    return matrix_distances.T


def euclidean_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.sqrt(np.sum(feature_weights * (np.square(train_matrix[:, None, :] -
                                                                   test_matrix[None, :, :])),
                                      axis=-1))
    return matrix_distances.T


def camberra_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.sum(feature_weights * (np.abs(train_matrix[:, None, :] - test_matrix[None, :, :]) /
                                                 train_matrix[:, None, :] + test_matrix[None, :, :]), axis=-1)
    where_nans = np.isnan(matrix_distances)
    matrix_distances[where_nans] = 0
    return matrix_distances.T


"""
def manhattan_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros((test_matrix.shape[0], train_matrix.shape[0]))
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            abs_differences = np.dot(feature_weights, np.abs(q - instance))
            matrix_distances[idy][idx] += abs_differences  # np.sum(abs_differences)

    return matrix_distances
"""


"""
def euclidean_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros((test_matrix.shape[0], train_matrix.shape[0]))
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            temp_sum = np.dot(feature_weights, np.square(q - instance))
            matrix_distances[idy][idx] += np.sqrt(temp_sum)
    return matrix_distances
"""



"""
def camberra_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros((test_matrix.shape[0], train_matrix.shape[0]))
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            numerator = np.abs(q - instance)
            denominator = np.abs(q + instance)
            # print(denominator)
            matrix_distances[idy][idx] += np.dot(feature_weights, numerator / denominator)
    return matrix_distances

"""
