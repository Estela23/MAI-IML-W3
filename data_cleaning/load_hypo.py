import pandas as pd
import os
from data_cleaning import exploring
from sklearn.preprocessing import MinMaxScaler
from scipy.io.arff import loadarff


def load_and_clean_hypo(file_full_path):
    """gets a full hypothyroid path, loading and cleaning it"""
    df_data = exploring.parse_arff_file_to_df(file_full_path)
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    df_data = decode_nominal_string_variables_by_column_list(df_data, nominal_columns)
    df_data = df_data[df_data.isnull().sum(axis=1) < 3]

    df_data = _delete_sex_unknown(df_data)  # 1.2.1
    df_data = _delete_tbg_and_measure_check_columns(df_data)  # 2
    df_data = _fill_nans_with_mean_and_split(df_data)  # 4.1.1
    classes, df_data = _split_encode_and_normalize(df_data)  # 5 & 6

    return classes, df_data


def load_train_test_fold(dataset_path: str, num_fold: int):
    """input1 dataset_name should be datasets/hypothyroid or datasets/kropt
    input2 num_fold is the specific fold we want to load the data
     returns the train correspondent test data of a specific fold with their classes"""
    root_folder = exploring.get_project_root()
    dataset_path = root_folder.joinpath(dataset_path)
    files_list = os.listdir(dataset_path)
    train_file = files_list[num_fold * 2 + 1]
    test_file = files_list[num_fold * 2]
    train_classes, train_data = load_and_clean_hypo(dataset_path.joinpath(train_file))
    test_classes, test_data = load_and_clean_hypo(dataset_path.joinpath(test_file))
    return train_classes, train_data, test_classes, test_data


def _delete_sex_unknown(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data = df_data.loc[df_data.sex != '?']

    return df_data


def _delete_tbg_and_measure_check_columns(df_data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['TBG', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
    df_data.drop(columns_to_drop, axis=1, inplace=True)

    return df_data


def _fill_nans_with_mean_and_split(df_data: pd.DataFrame):
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    numerical_columns = [column for column in df_data.columns if column not in nominal_columns]
    df_data = _fill_mixed_data_nans_with_mean(df_data, numerical_columns, nominal_columns)

    return df_data


def _fill_mixed_data_nans_with_mean(data: pd.DataFrame, numerical_columns, nominal_columns) -> pd.DataFrame:
    data_nominal = data.loc[:, nominal_columns]
    data_numerical = data.loc[:, numerical_columns]
    data_numerical.fillna(data_numerical.mean(), inplace=True)
    data = data_nominal.join(data_numerical)

    return data


def _split_encode_and_normalize(df_data: pd.DataFrame):
    classes = df_data['Class']
    df_data.drop('Class', axis=1, inplace=True)
    scaler = MinMaxScaler()
    df_data = _one_hot_encode_nominal_data(df_data)
    df_data = scaler.fit_transform(df_data)

    return classes, df_data


def decode_nominal_string_variables_by_column_list(dataframe, columns) -> pd.DataFrame:
    """
    Decode nominal variables from byte strings to strings.
    """
    for column in columns:
        dataframe[column] = dataframe[column].str.decode('utf8')

    return dataframe


def _one_hot_encode_nominal_data(df_data: pd.DataFrame) -> pd.DataFrame:
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    df_data = pd.get_dummies(data=df_data, columns=nominal_columns)

    return df_data


def parse_arff_to_df_all_data(dataset_path: str):# TO DELETE
    root_folder = exploring.get_project_root()
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


def explore_hypo():
    df_data = parse_arff_to_df_all_data('datasets/hypothyroid')

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
    # some cleanning
    df_data = parse_arff_to_df_all_data('datasets/hypothyroid')
    df_data.drop(columns=['TBG'], inplace=True)
    # TGB Has no values, so we can drop that column.
    df_data.drop(columns=['TBG'], inplace=True)
    # We get rid of the row with age equals to 455.
    df_data.drop(df_data[df_data.age == 455].index, inplace=True)

# example - load fold 5
# train_classes, train_data, test_classes, test_data = load_train_test_fold('datasets/hypothyroid', 5)

