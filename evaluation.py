from data_cleaning import load_kropt, load_hypo
from KNN import knn
import numpy as np
import time
from distance_metric import manhattan_metric, euclidean_metric, camberra_metric
from policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from weighting_knn import equal_weight, info_gain, reliefF

# Run the desired algorithm and count the number of cases correctly classified,
# the number of cases incorrectly classified, and the problem-solving time.


def apply_model(distance_function, k, voting_function, weighting_function, train_data, test_data):
    start_time = time.time()
    model = knn.KNN(distance_function, k, voting_function, weighting_function)
    model.fit(train_data)
    predictions = model.predict(test_data)  # instance 0
    n_correct_class = sum((predictions == train_data[:, -1]).all())  # instance 0
    end_time = time.time()
    time_fold = end_time - start_time

    return n_correct_class, time_fold


def apply_model_on10folds(data_name, distance_function, k, voting_function, weighting_function):
    n_folds = 10
    correct_class = np.zeros((n_folds, 0))
    time_folds = np.zeros((n_folds, 0))
    for j in range(n_folds):
        if data_name == 'datasets/kropt':
            train_data, test_data = load_kropt.load_train_test_fold(data_name, j)
        else:
            train_data, test_data = load_hypo.load_train_test_fold(data_name, j)

        correct_class[j], time_folds[j] = apply_model(distance_function, k, voting_function, weighting_function,
                                                      train_data, test_data)

    # calculate average over 10 folds
    incorrect_class = len(
        correct_class) - correct_class  # broadcasting: number of instances minus the correctly classified
    correctly_classified = np.average(correct_class)
    incorrectly_classified = np.average(incorrect_class)
    accuracy_by_folds = correct_class / len(correct_class)
    accuracy_average = np.average(accuracy_by_folds)
    time_average = np.average(time_folds)

    return accuracy_average, correctly_classified, incorrectly_classified, time_average


def run_experiment(data_name):
    # define the knn we choose
    distance_function = [manhattan_metric, euclidean_metric, camberra_metric]
    k = [1, 3, 5, 7]
    voting_function = [majority_class, inverse_distance_weighted, sheppards_work]
    weighting_function = [equal_weight, info_gain, reliefF]
    file = open("results_knn.txt", "w")
    for ind_dist in range(len(distance_function)):
        for ind_k in range(len(k)):
            for ind_vote in range(len(voting_function)):
                for ind_weight in range(len(weighting_function)):
                    accuracy, correctly_classified, incorrectly_classified, time_res = \
                        apply_model_on10folds(data_name, distance_function[ind_dist], k[ind_k],
                                              voting_function[ind_vote], weighting_function[ind_weight])
                # write results to txt
                file.write('[' + distance_function[ind_dist].__name__ + ', ' + str(k[ind_k]) + ', ' +
                           voting_function[ind_vote].__name__ + ', ' + weighting_function[ind_weight].__name__ + ' ]')
                file.write("\tAccuracy:   " + str(accuracy) + "; correctly classified: " + str(correctly_classified) +
                           "; incorrectly classified: " + str(incorrectly_classified) + ";\tTime:   " + str(time_res) +
                           " \n")

    file.close()


run_experiment('datasets/kropt')
run_experiment('hypo/kropt')
