# now we have 2 combination: one is the global winner, other with combination of the best in isolation
from KNN.algorithms.distance_metrics import manhattan_metric, euclidean_metric, camberra_metric
from KNN.algorithms.policies_knn import majority_class, inverse_distance_weighted, sheppards_work
from KNN.algorithms.weighting_knn import equal_weight, info_gain, reliefF
from evaluation import run_experiment
from scipy.stats import friedmanchisquare
import time

hypo_global_winner = [camberra_metric, 3, majority_class, info_gain]
hypo_isolation_winner = [camberra_metric, 3, inverse_distance_weighted, info_gain]

data_to_use = "datasets/hypothyroid"
file_name_to_export = "test.txt"

# Options: 1, 3, 5, 7,
k = [3]
# Options: manhattan_metric, euclidean_metric, camberra_metric
distance_functions = [camberra_metric]
# Options: majority_class, inverse_distance_weighted, sheppards_work
voting_functions = [majority_class]
# Options: equal_weight, info_gain, reliefF
weighting_functions = [info_gain]

reduction_techniques = [None]  # new_FCNN_rule, ENN, ib2

run_experiment(data_to_use, 'tair_test.txt', distance_functions, k, voting_functions, weighting_functions,
               reduction_techniques)


def apply_friedman(data1, data2, data3, alpha = 0.05):# friedman
    stat, p = friedmanchisquare(data1, data2, data3)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret

    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
