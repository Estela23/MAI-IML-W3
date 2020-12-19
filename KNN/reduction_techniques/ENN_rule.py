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


def enn(data_to_fit, predictions):
#     classes = data_to_fit[:, -1]
    for idx in range(len(predictions)):
        if predictions[idx] != data_to_fit[:, -1]:
            data = np.delete(data, 1, idx)
#             classes = np.delete(classes, 0, idx)

#     subset = np.concatenate((data, classes), axis=1)
    return subset


train_data, test_data = load_hypo.load_train_test_fold('datasets/hypothyroid', 1)
model = KNN(euclidean_metric, 3, majority_class, equal_weight, verbose=False)
model.fit(train_data)
predictions = model.predict(train_data)
time_init = time.time()
subset = enn(train_data, predictions)
time_end = time.time()
print(np.shape(subset))
print(f"Time calculating ENN: {time_end - time_init}")
