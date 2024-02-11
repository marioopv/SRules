import operator
import statistics
from operator import itemgetter
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imodels import RuleFitClassifier
from lightgbm import LGBMClassifier
from rulecosi import RuleCOSIClassifier
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from SRules.SRules import SRules
from SRules.Tests.Utils.DatasetUtils import read_dataset

path = f'../..'

n_splits = 10
n_repeats = 3
sorting_method = ['target_accuracy']

min_accuracy_coefficient = [0.95]
p_significance = [0.95]


@staticmethod
def define_file_header():
    file_header = "chi_square_percent_point_function, scale_feature_coefficient, min_accuracy_coefficient, " \
                  "min_number_class_per_node, sorting_method" \
                  ", dataset_test_size, dataset_test_categorizable" \
                  ", number_of_rules, cobertura, recursive"
    file_header += ', ensemble_accuracy, ensemble_f1_score, ensemble_precision_score, ensemble_recall, ensemble_roc_auc_score'
    file_header += ', tree_accuracy, tree_f1_score, tree_precision_score, tree_recall_score, tree_roc_auc_score'
    file_header += ', RuleFit_accuracy, RuleFit_f1_score, RuleFit_precision_score, RuleFit_recall_score, RuleFit_roc_auc_score'
    file_header += ', RuleCOSI_accuracy, RuleCOSI_f1_score, RuleCOSI_precision_score, RuleCOSI_recall_score, RuleCOSI_roc_auc_score'
    file_header += ', rules_accuracy, rules_f1_score, rules_precision_score, rules_recall_score, rules_roc_auc_score'
    file_header += ', RuleFit_numrules, RuleCOSI_numrules, rules_numrules\n'
    return file_header


ordered_columns = [" tree_f1_score", " ensemble_f1_score",
                     " RuleFit_f1_score", " RuleFit_numrules",
                     " RuleCOSI_f1_score"," RuleCOSI_numrules",
                     " rules_f1_score", " rules_numrules"]

important_columns = [" tree_f1_score", " ensemble_f1_score", " RuleFit_f1_score", " RuleCOSI_f1_score", " rules_f1_score"]
rule_columns = [" RuleFit_numrules", " RuleCOSI_numrules", " rules_numrules"]


@staticmethod
def define_summary_header():
    file_header = "filename, Recursive, Method, Dataset, Coverage"
    file_header += ', DT F1, DT vs SR ttest F1'
    file_header += ', Ensemble F1, Ensemble vs SR p-value F1'
    file_header += ', RFIT F1, RFIT vs SR p-value F1'
    file_header += ', RFIT \# rl. num., RC vs SR p-value \# rl. num.'
    file_header += ', RC F1, RC vs SR ttest F1'
    file_header += ', RC \# rl. num., RC vs SR p-value \# rl. num.'
    file_header += ', SR F1'
    file_header += ', SR \# rl. num.'
    file_header += ', SR F1 x coverage \n'
    return file_header


@staticmethod
def define_test_structure(dataset_names, ensembles, minImp, minInsNode, recursive, use_shap, use_lime):
    for filename in dataset_names:
        X, y, dataset, target_value_name, pandas_dataset = read_dataset(filename, f'{path}/data/{filename}.csv')
        for ensemble in ensembles:
            for recur in recursive:
                for minImportance in minImp:
                    for min_accuracy in min_accuracy_coefficient:
                        for chi2 in p_significance:
                            for sorting in sorting_method:
                                joblib.Parallel(n_jobs=8)(
                                    joblib.delayed(GenerateResults_File)(X, use_shap, use_lime, chi2, dataset, ensemble, filename, minIns, min_accuracy, recur, minImportance, sorting, y)
                                    for minIns in minInsNode)

@staticmethod
def summary_tests(dataset_names, ensembles, minImp, minInsNode, recursive):
    for filename in dataset_names:
        file_writer = open(get_summary_filename(filename), "w")
        file_writer.write(define_summary_header())
        for recur in recursive:
            for ensemble in ensembles:
                for minImportance in minImp:
                    for min_accuracy in min_accuracy_coefficient:
                        for chi2 in p_significance:
                            for sorting in sorting_method:
                                for minIns in minInsNode:
                                    current_filename = get_filename(ensemble, filename, minImportance, minIns, min_accuracy, recur, chi2, sorting)
                                    dataset = pd.read_csv(current_filename)
                                    file_writer.write(file_summary(dataset, ensemble, current_filename, filename))
        file_writer.close()

@staticmethod
def summary_tests_by_model_general(dataset_names, ensembles, minImp, minInsNode, recursive):
    for ensemble in ensembles:
        li = []
        file_writer = open(get_summary_ensemble_filename_general(ensemble), "w")
        file_writer.write(define_summary_header())
        for filename in dataset_names:
            for recur in recursive:
                best_filename = best_model(ensemble, filename, minImp, minInsNode, min_accuracy_coefficient, recur, p_significance, sorting_method)
                dataset = pd.read_csv(best_filename)
                li.append(dataset)
        frame = pd.concat(li, axis=0, ignore_index=True)
        new_file_descriptor = f'{ensemble}'
        file_writer.write(file_summary(frame, ensemble, new_file_descriptor, new_file_descriptor))
        file_writer.close()
@staticmethod
def summary_tests_general(dataset_names, ensembles, minImp, minInsNode, recursive):
    for recur in recursive:
        li = []
        file_writer = open(get_summary_ensemble_filename_general_recursive(recur), "w")
        file_writer.write(define_summary_header())
        for filename in dataset_names:
            for ensemble in ensembles:
                best_filename = best_model(ensemble, filename, minImp, minInsNode, min_accuracy_coefficient, recur, p_significance, sorting_method)
                dataset = pd.read_csv(best_filename)
                li.append(dataset)
        frame = pd.concat(li, axis=0, ignore_index=True)
        new_file_descriptor = f'{recur}'
        file_writer.write(file_summary(frame, recur, new_file_descriptor, new_file_descriptor))
        file_writer.close()

@staticmethod
def summary_tests_by_model(dataset_names, ensembles, minImp, minInsNode, recursive):

    for recur in recursive:
        for ensemble in ensembles:
            li = []
            file_writer = open(get_summary_ensemble_filename(ensemble, recur), "w")
            file_writer.write(define_summary_header())
            for filename in dataset_names:
                best_filename = best_model(ensemble, filename, minImp, minInsNode, min_accuracy_coefficient, recur, p_significance, sorting_method)
                dataset = pd.read_csv(best_filename)
                li.append(dataset)
            frame = pd.concat(li, axis=0, ignore_index=True)
            new_file_descriptor = f'{ensemble}-{recur}'
            file_writer.write(file_summary(frame, ensemble, new_file_descriptor, new_file_descriptor))
            file_writer.close()


@staticmethod
def summary_tests_by_dataset(dataset_names, ensembles, minImp, minInsNode, recursive):
    for recur in recursive:
        for ensemble in ensembles:
            for filename in dataset_names:
                file_writer = open(get_summary_ensemble_filename_by_dataset(ensemble, recur, filename), "w")
                file_writer.write(define_summary_header())
                best_filename = best_model(ensemble, filename, minImp, minInsNode, min_accuracy_coefficient, recur, p_significance, sorting_method)
                dataset = pd.read_csv(best_filename)
                file_writer.write(file_summary(dataset, ensemble, best_filename, filename))
                file_writer.close()


@staticmethod
def best_model(ensemble, filename, minImp, minInsNode, min_accuracy_coefficient, recur, p_significance, sorting_method):
    lis = []
    for minImportance in minImp:
        for min_accuracy in min_accuracy_coefficient:
            for chi2 in p_significance:
                for sorting in sorting_method:
                    for minIns in minInsNode:
                        current_filename = get_filename(ensemble, filename, minImportance, minIns, min_accuracy, recur,chi2, sorting)
                        dataset = pd.read_csv(current_filename)
                        lis.append((current_filename, round(statistics.mean(dataset.loc[:, " cobertura"]) * statistics.mean(dataset.loc[:, " rules_f1_score"]) * 100, 5)))

    return max(lis, key=itemgetter(1))[0]

@staticmethod
def file_summary(dataset, ensemble, current_filename, dataset_filename):
    text = (f'{current_filename}'
            f', {dataset[" recursive"].iloc[0]}, {type(ensemble)}, {dataset_filename}'
            f', {round(statistics.mean(dataset[" cobertura"])*100, 2)}±{round(statistics.stdev(dataset[" cobertura"])*100, 2)}')
    # KPIS OF RULES
    dict_values = {}
    for column in important_columns:
        new_data = dataset.loc[:, column]
        description = f', {round(statistics.mean(new_data)*100, 2)}±{round(statistics.stdev(new_data)*100, 2)}'
        if column != " rules_f1_score":
            t_stat, p_value = stats.ttest_ind(dataset.loc[:, " rules_f1_score"], new_data)
            if p_value < 0.0001:
                description += ', <0.0001'
            else:
                description += f', {round(p_value, 4)}'
        dict_values[column] = description
    # LEN OF RULES
    for rule_column in rule_columns:
        data_rule = dataset.loc[:, rule_column]
        description = f', {round(statistics.mean(data_rule), 2)}±{round(statistics.stdev(data_rule), 2)}'
        if rule_column != " rules_numrules":
            t_stat, p_value = stats.ttest_ind(dataset.loc[:, " rules_numrules"], data_rule)
            if p_value < 0.0001:
                description += ', <0.0001'
            else:
                description += f', {round(p_value, 4)}'
        dict_values[rule_column] = description
    for ordered in ordered_columns:
        text += dict_values[ordered]
    # SCORE
    text += (f', {round(statistics.mean(dataset.loc[:, " cobertura"]) * statistics.mean(dataset.loc[:, " rules_f1_score"])*100, 2)}'
             f'±{round(statistics.stdev(dataset.loc[:, " cobertura"]) * statistics.stdev(dataset.loc[:, " rules_f1_score"])*100, 2)}\n')
    return text


@staticmethod
def GenerateResults_File(X, use_shap, use_lime, chi2, dataset, ensemble, filename, minIns, min_accuracy, recur, minImp, sorting, y):
    filename_reults = get_filename(ensemble, filename, minImp, minIns, min_accuracy, recur, chi2, sorting)
    print(filename_reults)
    file_writer = open(filename_reults, "w")
    file_writer.write(define_file_header())
    for train, test in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats).split(X, y):
        fold_experiment_result(X, use_shap, use_lime, chi2, dataset, ensemble, file_writer, minIns, min_accuracy, recur, minImp, sorting,
                               test, train, y)
    file_writer.close()


@staticmethod
def get_filename(ensemble, filename, minImp, minIns, min_accuracy, recur, p_significance, sorting):
    return f'{path}/Results/battery_{filename}_{type(ensemble)}_recur{recur}_p_significance{p_significance}_min_accuracy{min_accuracy}_minIns{minIns}_minImp{minImp}_sorting{sorting}.csv'


@staticmethod
def get_summary_filename(filename):
    return f'{path}/Results/battery_{filename}_summary.csv'
@staticmethod
def get_summary_ensemble_filename_by_dataset(ensemble, recursive, filename):
    return f'{path}/Results/{type(ensemble).__name__}/battery_{filename}_recursive{recursive}_summary.csv'
@staticmethod
def get_summary_ensemble_filename(ensemble, recursive):
    return f'{path}/Results/{type(ensemble).__name__}/battery_recursive{recursive}_summary.csv'
@staticmethod
def get_summary_ensemble_filename_general(ensemble):
    return f'{path}/Results/{type(ensemble).__name__}/battery_summary.csv'
@staticmethod
def get_summary_ensemble_filename_general_recursive(recur):
    return f'{path}/Results/Recursive{recur}/battery_summary.csv'


@staticmethod
def fold_experiment_result(X, use_shap, use_lime, chi2, dataset, ensemble, file_writer, minIns, min_accuracy, recursive, minImp, sorting,
                           test,
                           train, y):
    X_train = X.loc[train].to_numpy()
    y_train = y.loc[train].to_numpy()
    X_test = X.loc[test].to_numpy()
    y_test = y.loc[test].to_numpy()
    train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                        columns=list(dataset['feature_names']) + [dataset.target_names])

    custom_scorer = make_scorer(f1_score, greater_is_better=True)

    # TREE
    clf_tree = GridSearchCV(
        # Evaluates the performance of different groups of parameters for a model based on cross-validation.
        DecisionTreeClassifier(),
        {
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
        },  # dict of parameters.
        cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
        scoring=custom_scorer)
    clf_tree.fit(X_train, y_train)
    tree = clf_tree.best_estimator_
    y_pred_test_tree = tree.predict(X_test)


    if type(ensemble) == type(RandomForestClassifier()) or type(LGBMClassifier()) or type(XGBClassifier()):
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
            'n_jobs': [2]  # threads
        }
    if type(ensemble) == type(CatBoostClassifier()) or type(GradientBoostingClassifier()):
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
        }

    if type(ensemble) == type(AdaBoostClassifier()):
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
        }
    if type(ensemble) == type(MLPClassifier()):
        param_grid = {
            'solver': ['adam'],
            'learning_rate_init': [0.0001],
            'max_iter': [300],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001, 0.005],
            'early_stopping': [True, False]
        }
    # ENSEMBLE
    clf_ensemble = GridSearchCV(
        # Evaluates the performance of different groups of parameters for a model based on cross-validation.
        ensemble,
        param_grid,  # dict of parameters.
        cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
        scoring=custom_scorer)
    clf_ensemble.fit(X_train, y_train)
    best_ensemble = clf_ensemble.best_estimator_
    y_pred_test_ensemble = best_ensemble.predict(X_test)


    # RuleFit
    RuleFIT = RuleFitClassifier()
    RuleFIT.fit(X_train.astype(int), y_train.astype(int), feature_names=dataset.feature_names)
    y_pred_test_RuleFit = RuleFIT.predict(X_test.astype(int))


    # RULECOSI
    RuleCosiMethod = RuleCOSIClassifier(
        base_ensemble=best_ensemble,
        conf_threshold=0.9,
        cov_threshold=0.0,
        column_names=dataset.feature_names)
    RuleCosiMethod.fit(X_train, y_train)
    y_pred_test_rulecosi = RuleCosiMethod.predict(X_test)

    # SRules
    SRulesMethod = SRules(
        feature_names=dataset.feature_names,
        target_value_name=dataset.target_names,
        p_significance=chi2,
        minImp=minImp,
        min_accuracy_coefficient=min_accuracy,
        minInsNode=minIns,
        display_features=False,
        display_logs=False,
        recursive=recursive
    )
    SRulesMethod.fit(method=best_ensemble,
                       X_train=X_train,
                       y_train=y_train,
                       original_dataset=train_pandas_dataset,
                       use_shap=use_shap,
                       use_lime=use_lime)
    y_pred_test_rules = SRulesMethod.predict(X_test, sorting_method=sorting)
    line_results = generate_results(chi2, min_accuracy, minIns, SRulesMethod, RuleCosiMethod, RuleFIT,
                                    minImp,
                                    sorting,
                                    y_pred_test_RuleFit, y_pred_test_ensemble,
                                    y_pred_test_rules,
                                    y_pred_test_tree, y_test, y_pred_test_rulecosi, recursive)
    file_writer.write(line_results)



@staticmethod
def generate_results(chi2, min_accuracy, minIns, SRulesMethod, RuleCosiMethod, RuleFIT, minImp, sorting, y_pred_test_RuleFit,
                     y_pred_test_ensemble, y_pred_test_rules, y_pred_test_tree, y_test, y_pred_test_rulecosi,
                     recursive):
    if not y_pred_test_rules or len(y_pred_test_rules) == 0:
        return empty_restuls(chi2, min_accuracy, minIns, minImp, y_test, recursive)

    line_results = extract_unitary_results(chi2, min_accuracy, minIns, SRulesMethod, RuleCosiMethod, RuleFIT,
                                           minImp,
                                           sorting,
                                           y_pred_test_RuleFit, y_pred_test_ensemble,
                                           y_pred_test_rules,
                                           y_pred_test_tree, y_test, y_pred_test_rulecosi, recursive)
    return line_results


@staticmethod
def empty_restuls(chi2, min_accuracy, min_class, minImp, y_test, recursive):
    return f'{chi2}, {minImp}, {min_accuracy}, {min_class}, NaN, {len(y_test)}, NaN, 0, {recursive}' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN'\
           f',NaN, NaN, NaN\n'


@staticmethod
def extract_unitary_results(chi2, min_accuracy, min_class, SRulesMethod, RuleCosiMethod, RuleFIT, minImp, sorting, y_pred_test_RuleFit,
                            y_pred_test_ensemble, y_pred_test_rules, y_pred_test_tree, y_test, y_pred_test_rulecosi,
                            recursive):
    # TODO: DIVIDE METHODS
    # DATASET CATEGORIZABLES

    if type(y_pred_test_ensemble[0]) == type("False"):
        y_pred_test_ensemble = np.array(y_pred_test_ensemble, dtype=bool)

    np_array_rules = np.array(y_pred_test_rules)
    filter_indices = np.where(np_array_rules != None)[0]
    filtered_y_test = np.array(y_test)[filter_indices].astype('int64')
    filtered_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype('int64')
    filtered_y_pred_test_rulecosi = np.array(y_pred_test_rulecosi)[filter_indices].astype('int64')
    filtered_y_pred_test_tree = np.array(y_pred_test_tree)[filter_indices].astype('int64')
    filtered_y_pred_test_RuleFit = np.array(y_pred_test_RuleFit)[filter_indices].astype('int64')
    filtered_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')

    # ACCURACY
    ensemble_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_ensemble)
    rulecosi_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_rulecosi)
    tree_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_rules)
    # F1
    ensemble_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_ensemble)
    rulecosi_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_rulecosi)
    tree_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_rules)
    # Precision
    ensemble_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_ensemble)
    rulecosi_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_rulecosi)
    tree_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_rules)
    # Recall
    ensemble_recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)
    rulecosi_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_rulecosi)
    tree_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_rules)

    # ROC AUC

    try:
        ensemble_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_ensemble)
    except ValueError:
        ensemble_roc_auc_score = 0.0
    try:
        tree_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_tree)
    except ValueError:
        tree_roc_auc_score = 0.0
    try:
        RuleFit_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    except ValueError:
        RuleFit_roc_auc_score = 0.0
    try:
        rules_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_rules)
    except ValueError:
        rules_roc_auc_score = 0.0
    try:
        rulecosi_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_rulecosi)
    except ValueError:
        rulecosi_roc_auc_score = 0.0

    line_results = f'{chi2}, {minImp}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(filtered_y_test)}, {len(SRulesMethod.minimal_rules_)}, {len(filtered_y_pred_test_rules) / len(y_test)}, {recursive}'
    line_results += f', {ensemble_accuracy}, {ensemble_f1_score}, {ensemble_precision_score}, {ensemble_recall}, {ensemble_roc_auc_score}'
    line_results += f', {tree_accuracy}, {tree_f1_score}, {tree_precision_score}, {tree_recall_score}, {tree_roc_auc_score}'
    line_results += f', {RuleFit_accuracy}, {RuleFit_f1_score}, {RuleFit_precision_score}, {RuleFit_recall_score}, {RuleFit_roc_auc_score}'
    line_results += f', {rulecosi_accuracy}, {rulecosi_f1_score}, {rulecosi_precision_score}, {rulecosi_recall_score}, {rulecosi_roc_auc_score}'
    line_results += f', {rules_accuracy}, {rules_f1_score}, {rules_precision_score}, {rules_recall_score}, {rules_roc_auc_score}'    
    line_results += f', {len(RuleFIT.rules_)}, {len(RuleCosiMethod.simplified_ruleset_.rules)}, {len(SRulesMethod.minimal_rules_)}\n'
    return line_results


@staticmethod
def generate_scores(filtered_y_test, filtered_y_pred_test_ensemble):
    accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_ensemble)
    f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_ensemble)
    precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_ensemble)
    recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)

    try:
        roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_ensemble)
    except ValueError:
        roc_auc_score = 0.0

    return accuracy, f1_score, precision_score, recall, roc_auc_score
