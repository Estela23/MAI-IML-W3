import pandas as pd
import os
from data_cleaning import exploring, hypothyroid_cleaning
from scipy.io.arff import loadarff

pd.options.mode.chained_assignment = None  # default='warn'


def normalize(df_data, min_df, max_df, numeric_columns):
    new_vals = (df_data[numeric_columns] - min_df) / (max_df - min_df)
    df_data[numeric_columns] = new_vals

    return df_data


def numerical_min_max_calc(df_data, numeric_columns):
    min_df = df_data[numeric_columns].min(axis=0)
    max_df = df_data[numeric_columns].max(axis=0)

    return min_df, max_df


def load_train_test_fold(dataset_path: str, num_fold: int):
    """input1 dataset_name should be datasets/hypothyroid or datasets/kropt
    input2 num_fold is the specific fold we want to load the data
     returns the train and test data of a specific fold with their classes"""
    root_folder = exploring.get_project_root()
    dataset_path = root_folder.joinpath(dataset_path)
    files_list = os.listdir(dataset_path)
    train_file = files_list[num_fold * 2 + 1]
    test_file = files_list[num_fold * 2]
    temp_data = loadarff(dataset_path.joinpath(train_file))
    train_data = pd.DataFrame(temp_data[0])
    temp_data2 = loadarff(dataset_path.joinpath(test_file))
    test_data = pd.DataFrame(temp_data2[0])
    n_train = len(train_data)
    whole_data = pd.concat([train_data, test_data])
    classes, df_data, numeric_columns = hypothyroid_cleaning.clean_filling_sex_and_filling_nans(whole_data)
    min_df, max_df = numerical_min_max_calc(df_data, numeric_columns)
    classes_train = classes.iloc[0:n_train]
    X_train = df_data.iloc[0:n_train, :]
    X_train = normalize(X_train, min_df, max_df, numeric_columns)
    classes_test = classes.iloc[n_train:-1]
    X_test = df_data.iloc[n_train:-1, :]
    X_test = normalize(X_test, min_df, max_df, numeric_columns)

    return classes_train, X_train, classes_test, X_test


# example - load fold 5
classes_train, X_train, classes_test, X_test = load_train_test_fold('datasets/hypothyroid', 5)
print(len(X_train))
print(len(X_test))
print(len(classes_train))
print(len(classes_test))

