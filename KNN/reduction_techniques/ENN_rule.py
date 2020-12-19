import numpy as np
from KNN.algorithms.policies_knn import random_policy
from data_cleaning import load_kropt, load_hypo
from KNN.knn import KNN
import numpy as np
import time
from KNN.algorithms.distance_metrics import manhattan_metric, euclidean_metric, camberra_metric
from KNN.algorithms.policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from KNN.algorithms.weighting_knn import equal_weight, info_gain, reliefF
from evaluation import apply_model


def ENN(data_to_fit, knn):
    temp_voting = knn._voting_function  # saved for later
    knn._voting_function = random_policy  # define the needed policy: majority -> if equals - random selection
    subset = data_to_fit

    predictions = knn.predict(subset)
    idx_adjusted = 0
    for idx in range(len(predictions)):
        if predictions[idx] == subset[idx_adjusted, -1]:
            idx_adjusted = idx_adjusted + 1
        else:
            subset = np.delete(subset, (idx_adjusted), axis=0)

    knn._voting_function = temp_voting  # return the policy of the original knn
    return subset


"""k = 7
train_data, test_data = load_hypo.load_train_test_fold('datasets/hypothyroid', 1)
print(np.shape(train_data))
model = KNN(euclidean_metric, k, majority_class, equal_weight, verbose=False)
model.fit(train_data)
subset = ENN(train_data, model)
print(np.shape(subset))


correct_class, time_folds = apply_model(euclidean_metric, k, majority_class, equal_weight, train_data, test_data)
incorrect_class = len(test_data) - correct_class  # broadcasting: number of instances minus the correctly classified
correctly_classified = np.average(correct_class)
accuracy = correct_class / len(test_data)
print("accuracy original data: ", accuracy)
# print(accuracy)
print("time: ", time_folds)


correct_class, time_folds = apply_model(euclidean_metric, k, majority_class, equal_weight, subset, test_data)
incorrect_class = len(test_data) - correct_class  # broadcasting: number of instances minus the correctly classified
correctly_classified = np.average(correct_class)
accuracy = correct_class / len(test_data)
print("accuracy reduced data: ", accuracy)
# print(accuracy)
print("time: ", time_folds)
"""
