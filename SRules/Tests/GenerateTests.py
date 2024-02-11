from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from SRules.Tests.Utils import TestUtils

ensembles = [
    CatBoostClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    ## XGBClassifier(),
    ## LGBMClassifier(),
]
minImp = [0.05]
minInsNode = [5, 7, 10, 15, 20, 25, 35, 50]
recursive = [False, True]
dataset_names = [
    "divorce",
    "wisconsin",
    "SPECT",
    "salud-covid",
    "tic-tac-toe",
    "kr-vs-kp",
]

# TODO: select best del nuestro automáticamente...
TestUtils.define_test_structure(dataset_names, ensembles, minImp, minInsNode, recursive)
TestUtils.summary_tests(dataset_names, ensembles, minImp, minInsNode, recursive)
TestUtils.summary_tests_by_dataset(dataset_names, ensembles, minImp, minInsNode, recursive)
TestUtils.summary_tests_by_model(dataset_names, ensembles, minImp, minInsNode, recursive)
TestUtils.summary_tests_by_model_general(dataset_names, ensembles, minImp, minInsNode, recursive)
TestUtils.summary_tests_general(dataset_names, ensembles, minImp, minInsNode, recursive)
# todo: display best y luego mean de cada uno...