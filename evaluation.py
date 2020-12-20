import time
from KNN import knn
import numpy as np
from data_cleaning import load_kropt, load_hypo
from KNN.algorithms.distance_metrics import manhattan_metric, euclidean_metric, camberra_metric
from KNN.algorithms.policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from KNN.algorithms.weighting_knn import equal_weight, info_gain, reliefF
from KNN.reduction_techniques.FCNN_rule import new_FCNN_rule
from KNN.reduction_techniques.ENN_rule import ENN
from KNN.reduction_techniques.ib2 import ib2


def apply_model(distance_function, k, voting_function, weighting_function, reduction_technique, train_data, test_data):
    start_time = time.time()
    if reduction_technique is None:
        verbose = True
    else:
        verbose = False
    model = knn.KNN(distance_function, k, voting_function, weighting_function, reduction_technique, verbose)
    model.fit(train_data)
    predictions = model.predict(test_data)  # instance 0
    classes = list(test_data[:, -1])
    correct_class = [i for i in range(len(classes)) if predictions[i] == classes[i]]
    n_correct_class = len(correct_class)

    end_time = time.time()
    time_fold = end_time - start_time

    return n_correct_class, time_fold


def apply_model_on10folds(data_name, distance_function, k, voting_function, weighting_function, reduction_technique):
    n_folds = 10
    correct_class = np.zeros((n_folds))
    time_folds = np.zeros((n_folds))
    for j in range(n_folds):
        if data_name == 'datasets/kropt':
            train_data, test_data = load_kropt.load_train_test_fold(data_name, j)
        else:
            train_data, test_data = load_hypo.load_train_test_fold(data_name, j)

        correct_class[j], time_folds[j] = apply_model(distance_function, k, voting_function, weighting_function,
                                                      reduction_technique, train_data, test_data)

        print(f"Fold {j} finished...")

    # calculate average over 10 folds
    incorrect_class = len(test_data) - correct_class  # broadcasting: number of instances minus the correctly classified
    correctly_classified = np.average(correct_class)
    incorrectly_classified = np.average(incorrect_class)
    accuracy_by_folds = correct_class / len(test_data)
    accuracy_average = np.average(accuracy_by_folds)
    time_average = np.average(time_folds)

    return accuracy_average, correctly_classified, incorrectly_classified, time_average


def write_results(file, dist_function, selected_k, vote_function, weight_function, accuracy, correctly_classified,
                  incorrectly_classified, time_res):

    file.write(f"[{dist_function.__name__}, {str(selected_k)}, {vote_function.__name__}, {weight_function.__name__}]")
    file.write(f"\tAccuracy:   {accuracy}; correctly classified: {correctly_classified}; incorrectly classified: "
               f"{incorrectly_classified};\tTime:   {time_res} \n")


def print_results(dist_function, selected_k, vote_function, weight_function, accuracy, correctly_classified,
                  incorrectly_classified, time_res):
    print(f"[{dist_function.__name__}, {selected_k}, {vote_function.__name__}, {weight_function.__name__}]")
    print(f"\tAccuracy:   {accuracy}; correctly classified: {correctly_classified}; incorrectly classified: "
          f"{incorrectly_classified};\tTime:   {time_res} \n")


def run_experiment(data_name, file_name_to_export, distance_functions, k, voting_functions, weighting_functions,
                   reduction_techniques):

    file = open(file_name_to_export, "w")
    for dist_function in distance_functions:
        for selected_k in k:
            for vote_function in voting_functions:
                for weight_function in weighting_functions:
                    for red_technique in reduction_techniques:
                        accuracy, correctly_classified, incorrectly_classified, time_res = \
                            apply_model_on10folds(data_name, dist_function, selected_k, vote_function,
                                                  weight_function, red_technique)
                        # Write results to txt
                        write_results(file, dist_function, selected_k, vote_function, weight_function, accuracy,
                                      correctly_classified, incorrectly_classified, time_res)

                        print_results(dist_function, selected_k, vote_function, weight_function, accuracy,
                                      correctly_classified, incorrectly_classified, time_res)

    file.close()


if __name__ == '__main__':

    data_to_use = "datasets/kropt"
    file_name_to_export = "test.txt"

    # Options: 1, 3, 5, 7,
    k = [3]
    # Options: manhattan_metric, euclidean_metric, camberra_metric
    distance_functions = [manhattan_metric]
    # Options: majority_class, inverse_distance_weighted, sheppards_work
    voting_functions = [majority_class]
    # Options: equal_weight, info_gain, reliefF
    weighting_functions = [equal_weight]

    reduction_techniques = [None]  # new_FCNN_rule, ENN, ib2

    run_experiment(data_to_use, file_name_to_export, distance_functions, k, voting_functions, weighting_functions,
                   reduction_techniques)
