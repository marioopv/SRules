import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch


dataset_names = [
    "divorce",
    "kr-vs-kp",
    "australian",
    "criotherapy",
    "data_banknote_authentication",
    "haberman",
    "mammographic_masses",
    "segmentation",
    "tic-tac-toe",
    "wisconsin",
    "salud-covid",
    "SPECT",
    "ionosphere",
    "credit",
    "connect-4"
]

@staticmethod
def read_dataset(dataset_name, dataset_path_name):
    dataset = pd.read_csv(dataset_path_name)
    dataset = dataset.replace('?', 'unknown')
    dataset = dataset.dropna()
    dataset.columns = [sub.replace('%', '') for sub in dataset.columns]
    target_value_name = dataset.columns[-1]

    one_hot_encoding = False
    continuous_to_discrete = False
    replace_target_value = False
    list_of_continuous_columns = []
    number_of_divisions = 10
    match dataset_name:

        case "australian":
            list_of_continuous_columns = ["A2", "A3", "A7", "A13", "A14"]
            continuous_to_discrete = True
            one_hot_encoding = True
        case "criotherapy":
            list_of_continuous_columns = ["age", "Time", "Area"]
            number_of_divisions = 5
            one_hot_encoding = True
            continuous_to_discrete = True
        case "data_banknote_authentication":
            list_of_continuous_columns = ["variance", "skewness", "curtosis", "entropy"]
            number_of_divisions = 10
            one_hot_encoding = True
            continuous_to_discrete = True
        case "haberman":
            target_true = 1
            target_false = 2
            replace_target_value = True
            list_of_continuous_columns = ["age", "year", "nodes"]
            number_of_divisions = 5
            one_hot_encoding = True
            continuous_to_discrete = True
        case "mammographic_masses":
            one_hot_encoding = True
        case "segmentation":

            target_true = 'BRICKFACE'
            target_false = 'REMAINDER'

            list_of_continuous_columns = ["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "VEDGE-MEAN", "VEDGE-SD",
                                          "HEDGE-MEAN",
                                          "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN",
                                          "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN",
                                          "HUE-MEAN"]
            list_of_continuous_columns = [sub.replace('-', '_').replace(' ', '') for sub in list_of_continuous_columns]
            replace_target_value = True
            number_of_divisions = 5
            one_hot_encoding = True
            continuous_to_discrete = True
        case "tic-tac-toe":
            target_true = 'positive'
            target_false = 'negative'
            replace_target_value = True
            one_hot_encoding = True
        case "wisconsin":
            one_hot_encoding = True
        case "salud-covid":
            one_hot_encoding = False
            continuous_to_discrete = False
        case "divorce":
            one_hot_encoding = True
            continuous_to_discrete = False
        case "SPECT":
            one_hot_encoding = False
            continuous_to_discrete = False
        case "credit":
            one_hot_encoding = True
            list_of_continuous_columns = ["LIMIT_BAL", "AGE",
                                          "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                          "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
            number_of_divisions = 10
            continuous_to_discrete = True
        case "ionosphere":
            target_true = 'g'
            target_false = 'b'
            replace_target_value = True
            one_hot_encoding = True

            list_of_continuous_columns = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
                                          "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23",
                                          "F24", "F25", "F26", "F27", "F28", "F29", "F30", "F31", "F32", "F33", "F34"]
            number_of_divisions = 10
            continuous_to_discrete = True
        case "kr-vs-kp":
            target_true = 'won'
            target_false = 'nowin'
            replace_target_value = True
            one_hot_encoding = True
            continuous_to_discrete = False
        case "connect-4":
            target_true = 'win'
            target_false = 'nowin'
            replace_target_value = True
            one_hot_encoding = True
            continuous_to_discrete = False

    if replace_target_value:
        dataset[target_value_name] = dataset[target_value_name].map({target_false: 0, target_true: 1})
    dataset.columns = [sub.replace('-', '_').replace(' ', '').replace('class', 'target_value') for sub in
                       dataset.columns]
    target_value_name = dataset.columns[-1]
    feature_names = dataset.columns[0:-1]

    if continuous_to_discrete:
        dataset = continuous_to_discrete_column(dataset, list_of_continuous_columns, number_of_divisions)

    if one_hot_encoding:
        dataset, feature_names = one_hot_encode_dataframe(dataset, feature_names)

    for column in dataset.columns:
        dataset[column] = dataset[column].astype(bool)

    X = dataset[feature_names]
    y = dataset[target_value_name]

    dataset_base = Bunch(
        data=X.to_numpy(),
        target=y.to_numpy(),
        target_names=target_value_name,
        feature_names=X.columns
    )

    return X, y, dataset_base, target_value_name, dataset

@staticmethod
def clean_names(name_list):
    return [sub.replace('?', '_').replace('.', '_').replace('-', '_').replace(' ', '') for sub in name_list]


@staticmethod
def one_hot_encode_dataframe(data, feature_names):
    enc = OneHotEncoder(sparse_output=False)
    encoded_array = enc.fit_transform(data.loc[:, feature_names])
    encoded_feature_names = clean_names(enc.get_feature_names_out())
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)
    encoded_pandas_dataset = pd.concat([df_encoded, data], axis=1)
    encoded_pandas_dataset.drop(labels=feature_names, axis=1, inplace=True)
    return encoded_pandas_dataset, encoded_feature_names
@staticmethod
def continuous_to_discrete_column(dataset, list_of_continuous_columns, number_of_divisions=10):
    labels = []

    if number_of_divisions == 10:
        labels = ['L_VeryLow', 'L_Low', 'L_Medium', 'L_High', 'L_VeryHigh',
                  'R_VeryLow', 'R_Low', 'R_Medium', 'R_High', 'R_VeryHigh']
    if number_of_divisions == 5:
        labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']

    for column_name in list_of_continuous_columns:
        dataset[column_name] = pd.cut(
            dataset[column_name]
            , number_of_divisions
            , labels=labels
        )

    return dataset
@staticmethod
def new_datasets(X_train, y_train, y_pred_train_rules, dataset):
    # Get indexes
    filter_indices = np.where(np.array(y_pred_train_rules) == None)[0]

    # new X_train
    np_filtered_X_train = np.array(X_train)[filter_indices]
    X_train = np_filtered_X_train
    np_filtered_y_train = np.array(y_train)[filter_indices]
    y_train = np_filtered_y_train

    # new Pandas dataset
    df = dataset.filter(items=filter_indices, axis=0)
    dataset = df

    new_len = len(X_train)
    return X_train, y_train, dataset, new_len