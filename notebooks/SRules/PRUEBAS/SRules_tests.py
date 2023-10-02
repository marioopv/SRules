from notebooks.SRules.test_utils import generate_results
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from notebooks.SRules.read_datasets import dataset_names, read_dataset



path = f'../../..'

results_file_name = f'{path}/Results'
test_size = 0.2
classifiers = [
    #CatBoostClassifier(),
    #GradientBoostingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    # XGBClassifier(),
    # LGBMClassifier(),
]

n_splits = 10
n_repeats = 3
# Different values
# TIME CONSUMING
# criterion = ["gini", "entropy", "log_loss"]
criterion = ["gini"]
scale_feature_coefficient = [0.05]
min_number_class_per_node_list = [5, 10, 20, 25, 30, 50]

# NOT TIME CONSUMING
min_accuracy_coefficient = [0.95]
chi_square_percent_point_function = [0.95]
sorting_method = ['target_accuracy']

recursive = [False, True]


dataset_names = [
    #"kr-vs-kp",
    #"divorce",
    #"SPECT",
    #"tic-tac-toe",
    #"wisconsin",
    "salud-covid",
]

for filename in dataset_names:
    dataset_path_name = f'{path}/data/{filename}.csv'
    X, y, dataset, target_value_name, pandas_dataset = read_dataset(filename, dataset_path_name)
    for classifier in classifiers:
        for recur in recursive:
            filename_reults = f'{results_file_name}/battery_{filename}_{classifier}_{recur}.csv'
            print(filename_reults)
            generate_results(recur, classifier, filename_reults, X, y, dataset, test_size,
                             chi_square_percent_point_function,
                             scale_feature_coefficient,
                             min_accuracy_coefficient,
                             min_number_class_per_node_list,
                             sorting_method, criterion, n_splits, n_repeats)