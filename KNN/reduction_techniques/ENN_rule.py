from KNN.algorithms.policies_knn import random_policy
import numpy as np
import math


def first_factor(number):
    i = 2
    while i <= int(math.sqrt(number)):
        if number % i == 0:
            return i
        else:
            i += 1


def ENN(data_to_fit, knn):
    temp_voting = knn._voting_function  # saved for later
    knn._voting_function = random_policy  # define the needed policy: majority -> if equals - random selection
    knn._data = data_to_fit
    subset_backup = data_to_fit
    # first_factor_number = first_factor(data_to_fit.shape[0])
    subsets = [data_to_fit[:7000, :], data_to_fit[7000:14000, :], data_to_fit[14000:21000, :], data_to_fit[21000:, :]]
    predictions = np.array([])
    for subset in subsets:
        predictions = np.append(predictions, knn.predict(subset))
    # predictions = knn.predict(subset_backup)
    idx_adjusted = 0
    for idx in range(len(predictions)):
        if predictions[idx] == subset_backup[idx_adjusted, -1]:
            idx_adjusted = idx_adjusted + 1
        else:
            subset_backup = np.delete(subset_backup, (idx_adjusted), axis=0)

    knn._voting_function = temp_voting  # return the policy of the original knn
    return subset_backup
