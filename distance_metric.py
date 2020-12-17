# File where the three optional distances will be implemented
import numpy as np


def distances_metrics(train_matrix, test_matrix, metric, feature_weights):
    if metric == "manhattan":
        matrix_distances = manhattan_metric(train_matrix, test_matrix, feature_weights)
    if metric == "euclidean":
        matrix_distances = euclidean_metric(train_matrix, test_matrix, feature_weights)
    if metric == "camberra":
        matrix_distances = camberra_metric(train_matrix, test_matrix, feature_weights)
    return matrix_distances

"""
def manhattan_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros((test_matrix.shape[0], train_matrix.shape[0]))
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            abs_differences = np.dot(feature_weights, np.abs(q - instance))
            matrix_distances[idy][idx] += abs_differences  # np.sum(abs_differences)

    return matrix_distances
"""


def manhattan_metric(train_matrix, test_matrix, feature_weights):

    #matrix_distances = np.abs(train_matrix[:, 0, None] - test_matrix[:, 0]) + \
    #                   np.abs(train_matrix[:, 1, None] - test_matrix[:, 1])

    #matrix_distances = np.dot(feature_weights, np.abs(train_matrix[:, None] - test_matrix).sum(-1))
    matrix_distances = np.sum(feature_weights * (np.abs(train_matrix[:, None, :] - test_matrix[None, :, :])), axis=-1)
    return matrix_distances.T


def euclidean_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros(test_matrix.shape[0], train_matrix.shape[0])
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            temp_sum = np.dot(feature_weights, np.square(q - instance))
            matrix_distances[idy][idx] += np.sqrt(temp_sum)
    return matrix_distances


def camberra_metric(train_matrix, test_matrix, feature_weights):
    matrix_distances = np.zeros(test_matrix.shape[0], train_matrix.shape[0])
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            numerator = np.abs(q - instance)
            denominator = np.abs(q + instance)
            matrix_distances[idy][idx] += np.dot(feature_weights, np.sum(numerator / denominator))
    return matrix_distances
