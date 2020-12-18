import numpy as np


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
    return np.array(centroids)


def FCNN_rule(train_data):
    centroids = compute_centroids(train_data)
    delta_S = centroids

    return reduced_train_data
