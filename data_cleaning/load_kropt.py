from data_cleaning import exploring
import os


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
        print(col)
        if 'col' in col:
            df_data[col] = [letter_to_labels(item)/8 for item in df_data[col]]
        else:
            df_data[col] = [int_to_label(item)/8 for item in df_data[col]]

    return classes, df_data


def letter_to_labels(letter):
    return ord(letter)-ord('a')


def int_to_label(integer):
    return ord(integer)-ord('1')


def load_train_test_fold(dataset_path: str, num_fold: int):
    """input1 dataset_name should be datasets/hypothyroid or datasets/kropt
    input2 num_fold is the specific fold we want to load the data
     returns the train correspondent test data of a specific fold with their classes"""
    root_folder = exploring.get_project_root()
    dataset_path = root_folder.joinpath(dataset_path)
    files_list = os.listdir(dataset_path)
    train_file = files_list[num_fold * 2]
    test_file = files_list[num_fold * 2 + 1]
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

