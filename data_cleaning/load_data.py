from scipy.io.arff import loadarff
import pandas as pd
from pathlib import Path
import os
from data_cleaning import exploring


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def parse_arff_to_matrix(dataset_path: str):
    root_folder = get_project_root()
    full_path = root_folder.joinpath(dataset_path)
    files_list = os.listdir(full_path)
    data = loadarff(full_path.joinpath(files_list[0]))
    matrix_df = pd.DataFrame(data[0])

    for idx in range(1, len(files_list)):
        file_to_open = full_path.joinpath(files_list[idx])
        data = loadarff(file_to_open)
        curr_df_data = pd.DataFrame(data[0])
        matrix_df = matrix_df.append(curr_df_data, ignore_index=False)

    return matrix_df


def load_and_clean_hypo():
    df_data = parse_arff_to_matrix('datasets/hypothyroid')
    df_data.drop(columns=['TBG'], inplace=True)
    # TGB Has no values, so we can drop that column.
    df_data.drop(columns=['TBG'], inplace=True)
    # We get rid of the row with age equals to 455.
    df_data.drop(df_data[df_data.age == 455].index, inplace=True)
    return df_data


def load_and_clean_kropt():
    df_data = parse_arff_to_matrix('datasets/kropt')
    return df_data


def explore_hypo():
    df_data = parse_arff_to_matrix('datasets/hypothyroid')

    print(f"There is NaN values: {df_data.isnull().values.any()}")
    print(f"Number of NaN values: {df_data.isnull().sum().sum()}")
    print(f"We can find NaN values in the following attributes: {df_data.columns[df_data.isna().any()].tolist()}")

    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    print(f"Nominal Columns: {nominal_columns}")
    numerical_columns = [column for column in df_data.columns if column not in nominal_columns]
    print(f"Numerical Columns: {numerical_columns}")

    # We can see a bit of the dataset
    print(df_data.head())

    # Exploring data
    exploring.print_unique_values_from_list_of_columns(df_data, nominal_columns)
    exploring.print_value_counts_from_list_of_columns(df_data, nominal_columns)
