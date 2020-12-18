import os
import sys
import time
import numpy as np


class KNN:

    def __init__(self, distance_function, k, voting_function, weighting_function, reductionKNNAlgorithm=None,
                 verbose=True):
        """
        distance : distance function to use
        k : number of nearest neighbours
        voting : voting function to determine class
        weighing : weighing function to calculate distances
        """
        self._data = []  # We will store here the training data
        self._distance_function = distance_function
        self._k = k
        self._voting_function = voting_function
        self._weighting_function = weighting_function
        self._reductionKNNAlgorithm = reductionKNNAlgorithm
        self._verbose = verbose

    def fit(self, data_to_fit):
        # We store the training data
        if self._reductionKNNAlgorithm is not None:
            data_to_fit = self._reductionKNNAlgorithm(data_to_fit=data_to_fit, knn=self)
        self._data = data_to_fit

    def predict(self, data_to_predict):
        if not self._verbose:
            self._block_print()
        # We calculate distances among all instances
        # Note that distance function will receive as 3rd argument the function for calculate the weighting
        # weights = self._weighting_function(self._data[:, :-1], self._data[:, -1], self._k)
        print("Calculating weigths...")
        time_init = time.time()
        weights = self._weighting_function(self._data[:, :-1], self._data[:, -1], self._k)
        time_end = time.time()
        print(f"Time calculating weights: {time_end-time_init}")
        print("Calculating distances...")
        time_init = time.time()
        distances = self._distance_function(self._data[:, :-1], data_to_predict[:, :-1], weights)
        time_end = time.time()
        print(f"Time calculating distances: {time_end-time_init}")
        # distances = np.sum(np.dot(weights, deltas))

        labels = self._data[:, -1]

        print("Sorting distances...")
        # We get the closest distances with its labels
        time_init = time.time()
        k_closests_distances, k_closest_labels = self._sort_distances_and_labels(distances, labels, self._k)
        time_end = time.time()
        print(f"Time sorting distances: {time_end-time_init}")

        print("Performing voting...")
        # We perform voting
        time_init = time.time()
        class_result = self._voting_function(k_closests_distances, k_closest_labels)
        time_end = time.time()
        print(f"Time performing votes: {time_end-time_init}")

        if not self._verbose:
            self._enable_print()

        return class_result

    def _sort_distances_and_labels(self, data, labels, k):
        labels = labels[np.newaxis, ...]
        index_array = np.argpartition(data, kth=k, axis=-1)
        k_closests_partition = np.take_along_axis(data, index_array, axis=-1)
        k_closests = k_closests_partition[:, :k]
        k_labels_partition = np.take_along_axis(labels, index_array, axis=-1)
        k_labels = k_labels_partition[:, :k]
        k_closests, k_labels = (list(t) for t in zip(*sorted(zip(k_closests.tolist(), k_labels.tolist()))))

        return k_closests, k_labels

    # Disable
    def _block_print(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def _enable_print(self):
        sys.stdout = sys.__stdout__
