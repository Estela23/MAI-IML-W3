import time
import numpy as np


def ib2(data_to_fit, knn, **_kwargs):
    subset = data_to_fit[:knn._k+1, :]

    for instance in data_to_fit:
        knn._data = subset  # TODO: check this is correct
        class_results = knn.predict(instance[None, :])
        if class_results == instance[-1]:
            subset = np.vstack((subset, instance))

    return subset
