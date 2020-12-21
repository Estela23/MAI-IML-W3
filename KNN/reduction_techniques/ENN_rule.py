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
    print(f"AFTER PERFORMING ENN REDUCTION, WE WENT FROM {data_to_fit.shape} to {subset_backup.shape}")
    return subset_backup


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
