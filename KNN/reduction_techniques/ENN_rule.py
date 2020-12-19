import numpy as np
from KNN.knn import KNN
import time
from data_cleaning import load_hypo
from evaluation import apply_model
from KNN.algorithms.distance_metrics import manhattan_metric, euclidean_metric, camberra_metric
from KNN.algorithms.policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from KNN.algorithms.weighting_knn import equal_weight, info_gain, reliefF



def ENN(data_to_fit, predictions):
    subset = data_to_fit
    idx_adjusted = 0
    for idx in range(len(predictions)):
        if predictions[idx] == subset[idx_adjusted, -1]:
            idx_adjusted = idx_adjusted + 1
        else:
            subset = np.delete(subset, (idx_adjusted), axis = 0)

    return subset


"""
train_data, test_data = load_hypo.load_train_test_fold('datasets/hypothyroid', 1)
print(np.shape(train_data))
model = KNN(euclidean_metric, 1, majority_class, equal_weight, verbose=False)
model.fit(train_data)
predictions = model.predict(train_data)
subset = ENN(train_data, predictions)
print(np.shape(subset))


correct_class, time_folds = apply_model(euclidean_metric, 1, majority_class, equal_weight, subset, test_data)
incorrect_class = len(test_data) - correct_class  # broadcasting: number of instances minus the correctly classified
correctly_classified = np.average(correct_class)
accuracy = correct_class / len(test_data)
print(correct_class)
print(accuracy)
print(time_folds)
"""
