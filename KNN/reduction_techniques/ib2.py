import time
import numpy as np


def ib2(train_data, knn):
    subset = train_data[:knn._k+1, :]

    for instance in train_data:
        knn.fit(subset)
        class_results = knn.predict(instance[None, :])
        if class_results == instance[-1]:
            subset = np.vstack((subset, instance))

    return subset

"""
train_data, test_data = load_kropt.load_train_test_fold('datasets/kropt', 1)
knn_alg = KNN(euclidean_metric, 3, majority_class, equal_weight, verbose=False)
time_init = time.time()
subset = ib2(train_data, knn_alg)
time_end = time.time()
print(f"Time calculating ib2: {time_end - time_init}")
"""