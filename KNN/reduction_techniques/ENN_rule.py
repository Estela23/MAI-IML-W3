import numpy as np
from KNN import knn
import time
from distance_metric import euclidean_metric
from policies_knn import majority_class
from weighting_knn import equal_weight
from data_cleaning import load_kropt, load_hypo


def enn(data_to_fit, knn):
    classes = data_to_fit[:, -1]
    data = data_to_fit[:, 0:-2]
    for idx, instance in enumerate(data):
        class_results = knn.predict(instance)
        if class_results != classes[idx]:
            data = np.delete(data, 1, idx)
            classes = np.delete(classes, 0, idx)

    subset = np.concatenate((data, classes), axis=1)
    return subset



""""""
train_data, test_data = load_kropt.load_train_test_fold('datasets/hypothyroid', 1)
knn_alg = knn(euclidean_metric, 3, majority_class, equal_weight, verbose=False)
time_init = time.time()
subset = enn(train_data)
time_end = time.time()
print(f"Time calculating ENN: {time_end - time_init}")
""""""