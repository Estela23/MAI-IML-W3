# now we have 2 combination: one is the global winner, other with combination of the best in isolation
from KNN.algorithms.distance_metrics import manhattan_metric, euclidean_metric, camberra_metric
from KNN.algorithms.policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from KNN.algorithms.weighting_knn import equal_weight, info_gain, reliefF
from evaluation import apply_model
import time

hypo_global_winner = [manhattan_metric, 7, sheppards_work, reliefF]
hypo_isolation_winner = [euclidean_metric, 7, majority_class, equal_weight]


def apply_model(distance_function, k, voting_function, weighting_function, train_data, test_data):
    start_time = time.time()
    model = knn.KNN(distance_function, k, voting_function, weighting_function, verbose=False)
    model.fit(train_data)
    predictions = model.predict(test_data)  # instance 0
    classes = list(test_data[:, -1])
    correct_class = [i for i in range(len(classes)) if predictions[i] == classes[i]]
    n_correct_class = len(correct_class)
    # n_correct_class = np.sum((predictions == classes).all())  # instance 0
    end_time = time.time()
    time_fold = end_time - start_time

    return n_correct_class, time_fold

