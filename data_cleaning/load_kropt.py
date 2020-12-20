from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from data_cleaning import exploring
import os
import numpy as np
import pandas as pd


def load_and_clean_kropt(file_full_path_train, file_full_path_test):
    """gets a full kropt path, loading and cleaning it"""
    df_data_train = exploring.parse_arff_file_to_df(file_full_path_train)
    df_data_test = exploring.parse_arff_file_to_df(file_full_path_test)
    train_samples = len(df_data_train)
    all_data = df_data_train.append(df_data_test, ignore_index=True)
    all_data = decode_nominal_string_variables_by_column_list(all_data, all_data[:])
    # We store the class information for future references
    # classes = df_data["game"]
    # all_data["game"] = pd.factorize(all_data["game"])[0] + 1 todo here
    # print(df_data.head())
    # We get rid of class information.
    # df_data = df_data.drop("game", axis=1)

    # label encoding the data
    # in case it is a column of chess - letter converted to int, else an integer is converted to int
    # division by 8 to normalize to 0-1
    # for col in all_data[:]: todo here
    #     if 'col' in col:
    #         all_data[col] = [letter_to_labels(item) / 8 for item in all_data[col]]
    #     elif 'row' in col:
    #         all_data[col] = [int_to_label(item) / 8 for item in all_data[col]]

    # label encoding the data
    le = LabelEncoder()
    for col in all_data[:]:
        all_data[col] = le.fit_transform(all_data[col])

    scaler = MinMaxScaler()
    all_data.iloc[:, :-1] = pd.DataFrame(scaler.fit_transform(all_data.iloc[:, :-1]), columns=all_data.columns[:-1])

    df_data_train = all_data.iloc[:train_samples, :]
    df_data_test = all_data.iloc[train_samples:, :]

    return df_data_train.to_numpy(), df_data_test.to_numpy()


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
    train_data, test_data = load_and_clean_kropt(dataset_path.joinpath(train_file), dataset_path.joinpath(test_file))

    return train_data, test_data


def decode_nominal_string_variables_by_column_list(dataframe, columns):
    """
    Decode nominal variables from byte strings to strings.
    """
    for column in columns:
        dataframe[column] = dataframe[column].str.decode('utf8')

    return dataframe


# example - load fold 1
# train_data, test_data = load_train_test_fold('datasets/kropt', 1)
# print(np.shape(train_data))
# print(np.shape(test_data))
# print(train_data[0:20, :])
