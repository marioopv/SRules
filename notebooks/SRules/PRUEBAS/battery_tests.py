from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from notebooks.SRules.read_datasets import dataset_names, read_dataset
from notebooks.SRules.test_utils import generate_battery_test

import warnings
warnings.filterwarnings("ignore")

# Different values
n_splits = 10
n_repeats = 3
chi_square_percent_point_function_list = [0.95, 0.97, 0.99]
scale_feature_coefficient_list = [0.05]
min_accuracy_coefficient_list = [0.95]
recursive = [True, False]
# min_number_class_per_node_list = [5, 7, 10]
sorting_method = "target_accuracy"


path = f'../../..'
results_file_name = f'{path}/Tests/battery_test_recursive_ALL_FINAL.csv'
# CONFIG

scale_feature_coefficient_list = [0.01]
chi_square_percent_point_function_list = [0.99]
min_accuracy_coefficient_list = [0.95]
min_number_class_per_node_list = [5, 10, 20, 25, 30, 50]

dataset_names = [
    "divorce",
    "tic-tac-toe",
    "wisconsin",
    "salud-covid",
    "SPECT",
    "kr-vs-kp",
]
# TODO:

classifiers = [
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    CatBoostClassifier(),
    RandomForestClassifier(),
    LGBMClassifier(),
    XGBClassifier()
]


f = open(results_file_name, "w")
file_header = f'Dataset, classifier, recursive, scorer, Coverage, DT, RF, RF+RFIT, RF+RFIT num rules, RF+RC, RF+RC num rules, RF+Rules, RF+Rules num rules\n'
print(file_header)
f.write(file_header)

for classifier in classifiers:
    for name in dataset_names:
        dataset_path_name = f'{path}/data/{name}.csv'
        X, y, dataset, target_value_name, pandas_dataset = read_dataset(name, dataset_path_name)
        for recur in recursive:
            for scale_feature_coefficient in scale_feature_coefficient_list:
                for chi_square_percent_point_function in chi_square_percent_point_function_list:
                    for min_number_class_per_node in min_number_class_per_node_list:
                        for min_accuracy_coefficient in min_accuracy_coefficient_list:
                            f_score, accuracy_score, precision_score, recall = generate_battery_test(recur, classifier, f, name, X, y, dataset,
                                                                                                     target_value_name, n_splits, n_repeats,
                                                                                                     chi_square_percent_point_function,
                                                                                                     scale_feature_coefficient,
                                                                                                     min_accuracy_coefficient,
                                                                                                     min_number_class_per_node,
                                                                                                     sorting_method)

f.close()
