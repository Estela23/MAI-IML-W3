import numpy as np
from scipy.spatial import distance
from data_cleaning import load_hypo

"""
def calculate_distance(x1, x2):
    return distance.cdist(np.array(x1), np.array(x2), metric="euclidean")
"""


def calculate_distance_vectors(x1, x2):
    if x1 is None or x2 is None:
        return float('inf')
    return np.linalg.norm(np.array(x1)-np.array(x2))


def compute_centroids(train_data):
    centroids = []
    labels = np.unique(train_data[:, -1])
    for i in range(len(labels)):
        instances_i = [train_data[j, :-1] for j in range(train_data.shape[0]) if train_data[j, -1] == i]
        temp_sum = np.zeros(train_data.shape[1] - 1)
        for j in range(len(instances_i)):
            temp_sum += instances_i[j]
        centroid_i = temp_sum / len(instances_i)
        centroids.append(centroid_i)
    return centroids


def FCNN_rule(train_data):
    T = train_data[:, :-1]
    S = np.empty((0, train_data.shape[1]-1))
    delta_S = compute_centroids(train_data)
    while len(delta_S) != 0:
        S = np.vstack((S, np.array(delta_S)))
        rep_p = [None] * S.shape[0]
        T_minus_S = [elem for elem in T if elem not in S]
        nearest_q = [None] * len(T_minus_S)
        for idx, q in enumerate(T_minus_S):
            for p in delta_S:
                if calculate_distance_vectors(nearest_q[idx], q) > calculate_distance_vectors(p, q):
                    nearest_q[idx] = p
            # TODO: arreglar estos índices del rep_p ...
            if train_data[q, -1] != train_data[nearest_q[idx], -1] and \
                    calculate_distance_vectors(nearest_q[idx], q) < calculate_distance_vectors(nearest_q[idx], rep_p[idx]):
                rep_p[nearest_q[idx]] = q
        delta_S = []
        for rep in rep_p:
            if rep is not None:
                delta_S.append(rep)
    return S


train_data, test_data = load_hypo.load_train_test_fold('datasets/hypothyroid', 1)

reduced_train_data = FCNN_rule(train_data)
