import numpy as np


def analyze(file_name):
    f = open(file_name, "r")
    txt = f.readlines()
    manhattan = []  # indexes of the line with manhattan
    euclidean = []
    camberra = []

    k1 = []
    k3 = []
    k5 = []
    k7 = []

    majority_class = []
    inverse_distance_weighted = []
    sheppards_work = []

    equal_weight = []
    info_gain = []
    reliefF = []
    for idx, line in enumerate(txt):
        current_params = line.strip('[]').split('\t')[0].split(',')
        #     for idx in range(len(current_params)):
        if current_params[0] == "manhattan_metric":
            manhattan.append(idx)
        elif current_params[0] == "euclidean_metric":
            euclidean.append(idx)
        else:
            camberra.append(idx)

        if int(current_params[1]) == 1:
            k1.append(idx)
        elif int(current_params[1]) == 3:
            k3.append(idx)
        elif int(current_params[1]) == 5:
            k5.append(idx)
        else:
            k7.append(idx)

        if current_params[2].strip(' ') == "majority_class":
            majority_class.append(idx)
        elif current_params[2].strip(' ') == "inverse_distance_weighted":
            inverse_distance_weighted.append(idx)
        else:
            sheppards_work.append(idx)
        if current_params[3].strip(' ]') == "equal_weight":
            equal_weight.append(idx)
        elif current_params[3].strip(' ]') == "info_gain":
            info_gain.append(idx)
        else:
            reliefF.append(idx)

    names = ["manhattan_metric", "euclidean", "camberra", "k1", "k3", "k5", "k7", "majority_class",
             "inverse_distance_weighted", "sheppards_work", "equal_weight", "info_gain", "reliefF"]
    for idx, lst in enumerate(
            [manhattan, euclidean, camberra, k1, k3, k5, k7, majority_class, inverse_distance_weighted, sheppards_work, \
             equal_weight, info_gain, reliefF]):
        print(names[idx])
        print_average_accuracy(lst, txt)

    best_idx = get_best_accuracy_idx(txt)
    print("best accuracy got " + get_combination(txt[best_idx]) + " with accuracy " + str(get_accuracy(txt[best_idx]))
          + " with time: " + str(get_time(txt[best_idx])))

    print("all euclidean \n")


def get_best_idx_for_param(txt, lst):
    best_idx_idx = get_best_accuracy_idx([print(txt[lst[i]]) for i in range(len(lst))])
    best_idx = lst[best_idx_idx]
    return best_idx


def get_best_accuracy_idx(txt):
    total_accuracy = []
    for line in txt:
        total_accuracy.append(get_accuracy(line))
    idx_max = total_accuracy.index(max(total_accuracy))
    return idx_max


def get_accuracy(line):
    return float(line.split(':')[1].split(';')[0])


def get_time(line):
    return float(line.split('Time: ')[1])


def print_average_accuracy(idx_lst, txt):
    accuracy_average = np.average([get_accuracy(txt[idx]) for idx in idx_lst])
    print(accuracy_average)


def get_combination(line):
    return line.split('\t')[0]


# analyze("results_knn_kropt_final_alberto.txt")
analyze("results_knn_hypo_final_alberto.txt")
