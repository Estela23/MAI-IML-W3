import numpy as np


def compute_centroids(data_to_fit, **_kwargs):
    centroids = []
    labels = np.unique(data_to_fit[:, -1])
    for i in range(len(labels)):
        instances_i = [data_to_fit[j, :-1] for j in range(data_to_fit.shape[0]) if data_to_fit[j, -1] == i]
        temp_sum = np.zeros(data_to_fit.shape[1] - 1)
        for j in range(len(instances_i)):
            temp_sum += instances_i[j]
        centroid_i = temp_sum / len(instances_i)
        centroids.append(centroid_i)
    return np.array(centroids)


def FCNN_rule(train_data):
    centroids = compute_centroids(train_data)
    delta_S = centroids

    return reduced_train_data
