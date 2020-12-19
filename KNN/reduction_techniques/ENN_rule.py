import numpy as np
from KNN.knn import KNN
import time
from KNN.algorithms.distance_metrics import euclidean_metric
from KNN.algorithms.policies_knn import majority_class
from KNN.algorithms.weighting_knn import equal_weight
from data_cleaning import load_hypo


# def enn(data_to_fit, knn):
#     classes = data_to_fit[:, -1]
#     data = data_to_fit[:, 0:-2]
#     for idx, instance in enumerate(data):
#         class_results = knn.predict(instance)
#         if class_results != classes[idx]:
#             data = np.delete(data, 1, idx)
#             classes = np.delete(classes, 0, idx)
#
#     subset = np.concatenate((data, classes), axis=1)
#     return subset


def ENN(data_to_fit, predictions):
    subset = data_to_fit
    idx_adjusted = 0
    for idx in range(len(predictions)):
        if predictions[idx] == subset[idx_adjusted, -1]:
            idx_adjusted = idx_adjusted + 1
        else:
            subset = np.delete(subset, (idx_adjusted), axis = 0)

    return subset


train_data, test_data = load_hypo.load_train_test_fold('datasets/hypothyroid', 1)
print(np.shape(train_data))
model = KNN(euclidean_metric, 3, majority_class, equal_weight, verbose=False)
model.fit(train_data)
predictions = model.predict(train_data)
subset = ENN(train_data, predictions)
print(np.shape(subset))
