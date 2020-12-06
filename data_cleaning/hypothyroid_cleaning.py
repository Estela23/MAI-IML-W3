from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import List
import pandas as pd
import random
from data_cleaning import exploring
import numpy as np


def clean_deleting_sex_and_delete_nans(df_data):
    df_data = _clean_first_nan_values_and_decode(df_data)  # 1.1
    df_data = _delete_sex_unknown(df_data)  # 1.2.1
    df_data = _delete_tbg_and_measure_check_columns(df_data)  # 2
    df_data = _clean_all_nans(df_data)  # 4.1.2
    classes, df_data = _split_class_encode_and_normalize(df_data)  # 5 & 6

    return classes, df_data


def clean_deleting_sex_and_filling_nans():
    df_data = _clean_first_nan_values_and_decode()  # 1.1
    df_data = _delete_sex_unknown(df_data)  # 1.2.1
    df_data = _delete_tbg_and_measure_check_columns(df_data)  # 2
    df_data = _fill_nans_with_mean_and_split(df_data)  # 4.1.1
    classes, df_data = _split_class_encode_and_normalize(df_data)  # 5 & 6

    return classes, df_data


def clean_filling_sex_and_delete_nans():
    df_data = _clean_first_nan_values_and_decode()  # 1.1
    df_data = _fill_sex_unknown(df_data)  # 1.2.2
    df_data = _delete_tbg_and_measure_check_columns(df_data)  # 2
    df_data = _clean_all_nans(df_data)  # 4.1.2
    classes, df_data = _split_class_encode_and_normalize(df_data)  # 5 & 6

    return classes, df_data


def clean_filling_sex_and_filling_nans(df_data, n_train):
    df_data = _clean_first_nan_values_and_decode(df_data)  # 1.1
    df_data, num_drop_upper = _fill_sex_unknown(df_data, n_train)  # 1.2.2
    df_data = _delete_tbg_and_measure_check_columns(df_data)  # 2
    numerical_columns = df_data._get_numeric_data().columns
    df_data = _fill_nans_with_mean_and_split(df_data)  # 4.1.1
    classes, df_data = _split_class_encode_and_normalize(df_data)  # 5 & 6

    return classes, df_data, numerical_columns, num_drop_upper


def _clean_first_nan_values_and_decode(df_data) -> pd.DataFrame:
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    df_data = decode_nominal_string_variables_by_column_list(df_data, nominal_columns)

    return df_data


def _random_male_female(prob_female):
    result = random.choices(['F', 'M'], weights=[prob_female, 1 - prob_female])

    return result[0]


def _fill_sex_unknown(df_data: pd.DataFrame, n_divide) -> pd.DataFrame:
    num_females = df_data['sex'].value_counts()[0]
    num_males = df_data['sex'].value_counts()[1]
    prob_female = num_females / (num_females + num_males)
    # We have samples of pregnants with 73 years, which we assume is wrong.
    mask = (df_data['pregnant'] == 't') & (df_data['age'] > 55)
    num_drop_upper = sum(mask[0:n_divide])
    df_data.drop(df_data[mask].index, inplace=True)
    # df_data['pregnant'][mask] = 'f'
    df_data.reset_index(inplace=True, drop=True)
    # We make sure that all pregnants are woman
    mask2 = (df_data['pregnant'] == 't') & (df_data['sex'] == 'M')
    num_drop_upper = num_drop_upper + sum(mask2[0:n_divide])
    # df_data['pregnant'][mask2] = 'f'
    df_data.drop(df_data[mask2].index, inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    # df_data['sex'] = df_data.apply(lambda x: 'F' if x.sex == '?' and x.pregnant == 't' else x, axis=1)
    # We introduce random sex
    df_data['sex'] = df_data['sex'].map(lambda x: _random_male_female(prob_female) if x == '?' else x)

    return df_data, num_drop_upper


def _delete_sex_unknown(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data = df_data.loc[df_data.sex != '?']

    return df_data


def _delete_tbg_and_measure_check_columns(df_data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['TBG', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
    df_data.drop(columns_to_drop, axis=1, inplace=True)

    return df_data


def _shuffle_dataset(pd_data: pd.DataFrame) -> pd.DataFrame:
    pd_data = shuffle(pd_data)
    pd_data.reset_index(inplace=True, drop=True)

    return pd_data


def _clean_all_nans(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data.dropna(inplace=True)

    return df_data


def _fill_mixed_data_nans_with_mean(data: pd.DataFrame, numerical_columns: List, nominal_columns: List) -> pd.DataFrame:
    data_nominal = data.loc[:, nominal_columns]
    data_numerical = data.loc[:, numerical_columns]
    data_numerical.fillna(data_numerical.mean(), inplace=True)

    data = data_nominal.join(data_numerical)

    return data


def _split_in_train_test(df_data: pd.DataFrame):
    X = df_data.drop(columns=['Class'])
    y = df_data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    return X_train, X_test, y_train, y_test


def _fill_nans_with_mean_and_split(df_data: pd.DataFrame):
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    numerical_columns = [column for column in df_data.columns if column not in nominal_columns]

    df_data = _fill_mixed_data_nans_with_mean(df_data, numerical_columns, nominal_columns)

    return df_data


def _one_hot_encode_nominal_data(df_data: pd.DataFrame) -> pd.DataFrame:
    nominal_columns = exploring.get_list_non_numerical_columns_of_df(df_data)
    df_data = pd.get_dummies(data=df_data, columns=nominal_columns)

    return df_data


def _split_class_encode_and_normalize(df_data: pd.DataFrame):
    classes = df_data['Class']
    df_data.drop('Class', axis=1, inplace=True)
    df_data = _one_hot_encode_nominal_data(df_data)

    return classes, df_data


def decode_nominal_string_variables_by_column_list(dataframe: pd.DataFrame, columns: List) -> pd.DataFrame:
    """
    Decode nominal variables from byte strings to strings.
    """
    for column in columns:
        dataframe[column] = dataframe[column].str.decode('utf8')

    return dataframe
