from data_cleaning import load_kropt
from KNN import knn
import numpy as np
import time


# Run the desired algorithm and count the number of cases correctly classified,
# the number of cases incorrectly classified, and the problem-solving time.

def apply_model(distance_function, k, voting_function, weighting_function, train_classes, train_data, test_classes,
                test_data):
    start_time = time.time()
    model = knn.KNN(distance_function, k, voting_function, weighting_function)
    model.fit(train_data)  # shouldn't we put here the data and the classes?
    # for i in range(len(test_classes)):
    predictions = model.predict(test_data)  # instance 0
    n_correct_class = sum((predictions == train_classes).all())  # instance 0
    end_time = time.time()
    time_fold = end_time - start_time

    return n_correct_class, time_fold


def run_experiment():
    # define data for experiment
    data_name = 'datasets/kropt'
    # define the knn we choose
    distance_function = "euclidean"
    k = 5
    voting_function = "majority_class"
    weighting_function = "equal_weight"
    n_folds = 10
    correct_class = np.zeros((n_folds, 0))
    time_folds = np.zeros((n_folds, 0))
    for j in range(n_folds):
        train_classes, train_data, test_classes, test_data = load_kropt.load_train_test_fold(data_name,
                                                                                             j)  # 1 as example
        correct_class[j], time_folds[j] = apply_model(distance_function, k, voting_function, weighting_function,
                                                      train_classes, train_data, test_classes, test_data)

    # calculate average over 10 folds
    incorrect_class = len(test_classes) - correct_class
    accuracy = correct_class / len(test_classes)
    accuracy_average = np.average(correct_class)
    time_average = np.average(time_folds)
    # write results to txt
    file = open("results_knn.txt", "w")
    file.write("Correctly classified:   " + str(correct_class) + " \n")
    file.write("Incorrectly classified: " + str(incorrect_class) + " \n")
    file.write("Accuracy fold:  " + str(accuracy) + " \n")
    file.write("Accuracy average:   " + str(accuracy_average) + " \n")
    file.write("Time by fold:   " + str(time_folds) + " \n")
    file.write("Time average:   " + str(time_average) + " \n")
    file.close()


run_experiment()