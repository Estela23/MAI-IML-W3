# File where the three optional distances will be implemented
import numpy as np


def distances_metrics(train_matrix, test_matrix, metric):
    if metric == "manhattan":
        matrix_delta = manhattan_metric(train_matrix, test_matrix)
    if metric == "euclidean":
        matrix_delta = euclidean_metric(train_matrix, test_matrix)
    if metric == "camberra":
        matrix_delta = camberra_metric(train_matrix, test_matrix)
    return matrix_delta


def manhattan_metric(train_matrix, test_matrix, feature_weights):
    matrix_delta = np.zeros((test_matrix.shape[0], train_matrix.shape[0]))
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            abs_differences = np.dot(feature_weights, np.abs(q - instance))
            matrix_delta[idy][idx] += abs_differences  # np.sum(abs_differences)
    return matrix_delta


def euclidean_metric(train_matrix, test_matrix, feature_weights):
    matrix_delta = np.zeros(test_matrix.shape[0], train_matrix.shape[0])
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            temp_sum = np.dot(feature_weights, np.square(q - instance))
            matrix_delta[idy][idx] += np.sqrt(temp_sum)
    return matrix_delta


def camberra_metric(train_matrix, test_matrix, feature_weights):
    matrix_delta = np.zeros(test_matrix.shape[0], train_matrix.shape[0])
    for idy, q in enumerate(test_matrix):
        for idx, instance in enumerate(train_matrix):
            numerator = np.abs(q - instance)
            denominator = np.abs(q + instance)
            matrix_delta[idy][idx] += np.dot(feature_weights, np.sum(numerator / denominator))
    return matrix_delta
