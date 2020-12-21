import numpy as np
from scipy.spatial import distance
from data_cleaning import load_hypo


def calculate_distance(x1, x2):
    return distance.cdist(np.array(x1), np.array(x2), metric="euclidean")


def assign_instance_to_subset(data, subset_data):
    """
    Assigns each instance to the closest member of the subset
    """
    distances = calculate_distance(data, subset_data)
    assigns = []
    for i in range(len(data)):
        assigns.append(np.argmin(distances[i]) + 1)
    return np.array(assigns)


def calculate_distance_vectors(x1, x2):
    return np.linalg.norm(np.array(x1)-np.array(x2))


def compute_centroids(data_to_fit):
    centroids = []
    labels_centroids = np.unique(data_to_fit[:, -1])
    for label in labels_centroids:
        instances_i = [data_to_fit[j, :-1] for j in range(data_to_fit.shape[0]) if data_to_fit[j, -1] == label]
        temp_sum = np.zeros(data_to_fit.shape[1] - 1)
        for j in range(len(instances_i)):
            temp_sum += instances_i[j]
        real_centroid_i = temp_sum / len(instances_i)
        distances_to_centroid = [calculate_distance_vectors(real_centroid_i, instances_i[j]) for j in range(len(instances_i))]
        index = distances_to_centroid.index(np.amin(np.array(distances_to_centroid)))
        centroid_i = instances_i[index]
        centroids.append(centroid_i)
    return centroids, labels_centroids


def FCNN_rule(data_to_fit, **_kwargs):
    centroids, labels_centroids = compute_centroids(data_to_fit)
    delta_S = True
    while delta_S:
        # Assign each instance to a member of the subset
        assigns = assign_instance_to_subset(data_to_fit[:, :-1], centroids)

        delta_S = []
        for i in range(len(centroids)):
            misclassified = []
            indexes = []
            for idx, instance in enumerate(data_to_fit):
                # If the instance is assigned at the member of the subset but is not of the same
                # category, add to the list of misclassified
                if assigns[idx] == i+1 and data_to_fit[idx, -1] != labels_centroids[i]:
                    misclassified.append(instance[:-1])
                    indexes.append(idx)
            if misclassified:
                # Get the representative misclassified instance from the set (in this case,
                # FCNN, takes the closest misclassified instance to the member of the subset)
                rep = np.argmin(calculate_distance(misclassified, np.array(centroids[i:i + 1])))
                delta_S.append(indexes[rep])

        # Actualize the new subset
        for i in delta_S:
            centroids = np.vstack((centroids, data_to_fit[i, :-1]))
            labels_centroids = np.hstack((labels_centroids, data_to_fit[i, -1]))

    reduced_data = np.hstack((centroids, labels_centroids[:, None]))
    return reduced_data
