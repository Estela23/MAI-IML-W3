from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.arff import loadarff
from pathlib import Path


from sklearn.metrics import classification_report


def get_list_non_numerical_columns_of_df(dataframe: pd.DataFrame) -> List:
    """
    Returns a list of column names, where this names corresponds to non numerical attributes.
    """
    cols = dataframe.columns
    num_cols = dataframe._get_numeric_data().columns

    return list(set(cols) - set(num_cols))


def print_unique_values_from_list_of_columns(dataframe: pd.DataFrame, columns: List):
    for column in columns:
        data = dataframe[column].unique()
        print(f"Number of different values of {column} column: {len(data)}")
        print(f"Different values of {column} column:\n{data}")


def print_value_counts_from_list_of_columns(dataframe: pd.DataFrame, columns: List):
    for column in columns:
        print(f"Counting values for {column} column:\n{dataframe[column].value_counts()}")


def print_nan_values_from_numerical_columns(dataframe: pd.DataFrame, numerical_columns: List):
    for column in numerical_columns:
        data = dataframe[column]
        print(f"NaN values in {column} column are: {data.isnull().sum().sum()}/{len(data.index)}")


def plot_value_counts_from_list_of_columns(dataframe: pd.DataFrame, columns: List):
    for column in columns:
        ax = sns.countplot(x=column, data=dataframe)
        plt.show()


def plot_value_counts_from_list_of_columns_same_plot(dataframe: pd.DataFrame, columns: List):
    fig, axs = plt.subplots(1, len(columns), figsize=(10, 5))
    for i, column in enumerate(columns):
        sns.countplot(x=column, data=dataframe, ax=axs[i])
    plt.show()


def plot_boxplot_from_numerical_columns(dataframe: pd.DataFrame, numerical_columns: List):
    for column in numerical_columns:
        ax = sns.boxplot(x=dataframe[column])
        plt.show()


def plot_confusion_matrix(confusion_matrix, labels):
    group_counts = ["{0: 0.0f}".format(value) for value in confusion_matrix.flatten()]
    group_percentages = ["{0: 0.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    square_info = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    square_info = np.asarray(square_info).reshape(len(confusion_matrix[0]), len(confusion_matrix[1]))
    sns.heatmap(confusion_matrix, annot=square_info, fmt="", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.show()


def parse_arff_file_to_df(dataset_path: str):
    data = loadarff(dataset_path)
    matrix_df = pd.DataFrame(data[0])

    return matrix_df


def get_project_root():
    return Path(__file__).parent.parent

