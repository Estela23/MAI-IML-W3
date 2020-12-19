# File where the three optional policies will be implemented
import numpy as np

"""
:param matrix_k_neighbours: list of lists, of the distances between each test data (row) and
                            its k nearest neighbours (sorted from min distance to max)
:param list_y_train: list of lists, in the first list we have the labels (strings)
                    of the k NN from the first element in the test set
:return: predicted_labels: list of length test set with the label predicted for each
"""


def majority_class(matrix_k_neighbours, list_y_train):
    # We iterate through all the instances to classify in the test set
    predicted_labels = []
    for n_instance, neighbours in enumerate(matrix_k_neighbours):
        labels_neighbours = list_y_train[n_instance]
        labels_dict = {}
        for label in set(labels_neighbours):
            count_label = labels_neighbours.count(label)
            labels_dict[label] = count_label
        possible_labels = [k for k, v in labels_dict.items() if v == labels_dict[max(labels_dict, key=labels_dict.get)]]
        if len(possible_labels) == 1:
            predicted_labels.append(possible_labels[0])
        else:
            for label in labels_neighbours:
                if label in possible_labels:
                    predicted = label
                    predicted_labels.append(predicted)
                    break

    return predicted_labels


def inverse_distance_weighted(matrix_k_neighbours, list_y_train):
    matrix_k_neighbours = np.array(matrix_k_neighbours)
    predicted_labels = []
    for n_instance, neighbours in enumerate(matrix_k_neighbours):
        labels_neighbours = list_y_train[n_instance]
        weights_neighbours = 1 / np.array(neighbours)
        labels_dict = {}
        for label in set(labels_neighbours):
            indexes = [i for i in range(len(labels_neighbours)) if labels_neighbours[i] == label]
            weight_label = np.sum(weights_neighbours[indexes])
            labels_dict[label] = weight_label
        possible_labels = [k for k, v in labels_dict.items() if v == labels_dict[max(labels_dict, key=labels_dict.get)]]
        if len(possible_labels) == 1:
            predicted_labels.append(possible_labels[0])
        else:
            for label in labels_neighbours:
                if label in possible_labels:
                    predicted = label
                    predicted_labels.append(predicted)
                    break
    return predicted_labels


def sheppards_work(matrix_k_neighbours, list_y_train):
    matrix_k_neighbours = np.array(matrix_k_neighbours)
    predicted_labels = []
    for n_instance, neighbours in enumerate(matrix_k_neighbours):
        labels_neighbours = list_y_train[n_instance]
        weights_neighbours = np.exp(-np.array(neighbours))
        labels_dict = {}
        for label in set(labels_neighbours):
            indexes = [i for i in range(len(labels_neighbours)) if labels_neighbours[i] == label]
            weight_label = np.sum(weights_neighbours[indexes])
            labels_dict[label] = weight_label
        possible_labels = [k for k, v in labels_dict.items() if v == labels_dict[max(labels_dict, key=labels_dict.get)]]
        if len(possible_labels) == 1:
            predicted_labels.append(possible_labels[0])
        else:
            for label in labels_neighbours:
                if label in possible_labels:
                    predicted = label
                    predicted_labels.append(predicted)
                    break

    return predicted_labels
