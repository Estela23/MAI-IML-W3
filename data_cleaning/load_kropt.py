from data_cleaning import exploring
import os
import numpy as np


def load_and_clean_kropt(file_full_path):
    """gets a full kropt path, loading and cleaning it"""
    df_data = exploring.parse_arff_file_to_df(file_full_path)
    df_data = decode_nominal_string_variables_by_column_list(df_data, df_data[:])
    # exploring.print_unique_values_from_list_of_columns(df_data, df_data[:])
    # exploring.print_value_counts_from_list_of_columns(df_data, df_data[:])
    # We store the class information for future references
    classes = df_data["game"]

    # We get rid of class information.
    df_data = df_data.drop("game", axis=1)

    # label encoding the data
    # in case it is a column of chess - letter converted to int, else an integer is converted to int
    # division by 8 to normalize to 0-1
    for col in df_data[:]:
        if 'col' in col:
            df_data[col] = [letter_to_labels(item) / 8 for item in df_data[col]]
        else:
            df_data[col] = [int_to_label(item) / 8 for item in df_data[col]]
    df_data = np.array(df_data)
    return classes, df_data


def cross_validation_kropt(dataset_path: str, val_fold_idx: int):
    """input1: 'datasets/kropt', folder name where data is found
    input 2: which fold to exclude from the data
    outpus: data and classes for both train and test for the 9 or 10 folds processed"""
    train_classes_lst = []
    train_data_lst = []
    test_classes_lst = []
    test_data_lst = []

    for i in range(10):
        if i == val_fold_idx:
            continue
        else:
            train_classes, train_data, test_classes, test_data = load_train_test_fold(dataset_path, i)
            train_classes_temp = train_classes.values.tolist()
            train_classes_lst.append(train_classes_temp)
            train_data_lst.append(train_data)
            test_classes_temp = test_classes.values.tolist()
            test_classes_lst.append(test_classes_temp)
            test_data_lst.append(test_data)

    flatten = lambda t: [item for sublist in t for item in sublist]
    train_classes_out = flatten(train_classes_lst)
    test_classes_out = flatten(test_classes_lst)
    train_data_temp = flatten(train_data_lst)
    test_data_temp = flatten(test_data_lst)

    train_data_arr = [np.asarray(item) for item in train_data_temp]
    test_data_arr = [np.asarray(item) for item in test_data_temp]

    return train_classes_out, train_data_arr, test_classes_out, test_data_arr


def letter_to_labels(letter):
    return ord(letter) - ord('a')


def int_to_label(integer):
    return ord(integer) - ord('1')


def load_train_test_fold(dataset_path: str, num_fold: int):
    """input1 dataset_name should be datasets/hypothyroid or datasets/kropt
    input2 num_fold is the specific fold we want to load the data
     returns the train correspondent test data of a specific fold with their classes"""
    root_folder = exploring.get_project_root()
    dataset_path = root_folder.joinpath(dataset_path)
    files_list = os.listdir(dataset_path)
    train_file = files_list[num_fold * 2 + 1]
    test_file = files_list[num_fold * 2]
    train_classes, train_data = load_and_clean_kropt(dataset_path.joinpath(train_file))
    test_classes, test_data = load_and_clean_kropt(dataset_path.joinpath(test_file))
    return train_classes, train_data, test_classes, test_data


def decode_nominal_string_variables_by_column_list(dataframe, columns):
    """
    Decode nominal variables from byte strings to strings.
    """
    for column in columns:
        dataframe[column] = dataframe[column].str.decode('utf8')

    return dataframe

# example - load fold 1
# train_classes, train_data, test_classes, test_data = load_train_test_fold('datasets/kropt', 1)

# example - cross val kropt - if we want all the data put more 10 or more in fold number to exclude
# train_classes, train_data, test_classes, test_data = cross_validation_kropt('datasets/kropt', 11)
# print(np.shape(train_classes))
# print(np.shape(train_data))
# print(np.shape(test_classes))
# print(np.shape(test_data))
