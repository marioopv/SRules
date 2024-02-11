import numpy as np
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imodels import RuleFitClassifier
from lightgbm import LGBMClassifier
from rulecosi import RuleCOSIClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from SRules.SRules import SRules


def generate_results(recur, classifier, results_file_name, X, y, dataset, test_size,
                     chi_square_percent_point_function,
                     scale_feature_coefficient,
                     min_accuracy_coefficient,
                     min_number_class_per_node,
                     sorting_method, criterion="gini", n_splits=10, n_repeats=3):
    # WRITE FILE
    f = open(results_file_name, "w")
    file_header = "ensemble_criterion, chi_square_percent_point_function, scale_feature_coefficient, min_accuracy_coefficient, " \
                  "min_number_class_per_node, sorting_method" \
                  ", dataset_test_size, dataset_test_categorizable" \
                  ", number_of_rules, cobertura"
    file_header += ', ensemble_accuracy, ensemble_f1_score, ensemble_precision_score, ensemble_recall, ensemble_roc_auc_score'
    file_header += ', tree_accuracy, tree_f1_score, tree_precision_score, tree_recall_score, tree_roc_auc_score'
    file_header += ', RuleFit_accuracy, RuleFit_f1_score, RuleFit_precision_score, RuleFit_recall_score, RuleFit_roc_auc_score'
    file_header += ', RuleCOSI_accuracy, RuleCOSI_f1_score, RuleCOSI_precision_score, RuleCOSI_recall_score, RuleCOSI_roc_auc_score'
    file_header += ', rules_accuracy, rules_f1_score, rules_precision_score, rules_recall_score, rules_roc_auc_score\n'

    print(file_header)
    f.write(file_header)

    repeated_kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    for train, test in repeated_kfold.split(X, y):
        X_train = X.loc[train].to_numpy()
        y_train = y.loc[train].to_numpy()
        X_test = X.loc[test].to_numpy()
        y_test = y.loc[test].to_numpy()
        for_results(recur, classifier, X_train, X_test, y_train, y_test, chi_square_percent_point_function, criterion,
                    dataset, f,
                    min_accuracy_coefficient,
                    min_number_class_per_node, scale_feature_coefficient, sorting_method, test_size)
    f.close()


def for_results(recur, classifier, X_train, X_test, y_train, y_test, chi_square_percent_point_function, criterion,
                dataset, f,
                min_accuracy_coefficient,
                min_number_class_per_node, scale_feature_coefficient, sorting_method, test_size):
    train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                        columns=list(dataset['feature_names']) + [dataset.target_names])
    X_train_int = X_train.astype(int)
    y_train_int = y_train.astype(int)
    X_test_int = X_test.astype(int)
    y_test_int = y_test.astype(int)

    print('Sizes (without target):')
    print(f'Original size {dataset.data.shape}')
    print(f'Train size {X_train.shape}')
    print(f'Test size {X_test.shape}')

    # TREE
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    y_pred_test_tree = tree.predict(X_test)
    # RuleFit
    ruleFit = RuleFitClassifier()
    ruleFit.fit(X_train_int, y_train_int, feature_names=dataset.feature_names)
    y_pred_test_RuleFit = ruleFit.predict(X_test_int)
    for criteria in criterion:
        generate_results_from_criterion(recur, classifier, X_test, X_train, chi_square_percent_point_function, criteria,
                                        dataset, f,
                                        min_accuracy_coefficient, min_number_class_per_node, scale_feature_coefficient,
                                        sorting_method, train_pandas_dataset, y_pred_test_RuleFit, y_pred_test_tree,
                                        y_test, y_train)


def generate_results_from_criterion(recur, classifier, X_test, X_train, chi_square_percent_point_function, criteria,
                                    dataset, f,
                                    min_accuracy_coefficient, min_number_class_per_node, scale_feature_coefficient,
                                    sorting_method, train_pandas_dataset, y_pred_test_RuleFit, y_pred_test_tree, y_test,
                                    y_train):
    # ENSEMBLE
    ensemble = classifier
    ensemble.fit(X_train, y_train)
    y_pred_test_ensemble = ensemble.predict(X_test)

    rulecosi = RuleCOSIClassifier(
        base_ensemble=ensemble,
        conf_threshold=0.9,
        cov_threshold=0.0,
        column_names=dataset.feature_names)
    rulecosi.fit(X_train, y_train)
    y_pred_test_rulecosi = rulecosi.predict(X_test)

    for scaler in scale_feature_coefficient:
        for min_class in min_number_class_per_node:
            for min_accuracy in min_accuracy_coefficient:
                for chi2 in chi_square_percent_point_function:
                    for sorting in sorting_method:
                        SubrogateRules = SRules(
                            feature_names=dataset.feature_names,
                            target_value_name=dataset.target_names,
                            p_significance=chi2,
                            scale_feature_coefficient=scaler,
                            min_accuracy_coefficient=min_accuracy,
                            minInsNode=min_class,
                            display_features=False,
                            display_logs=False,
                            recursive=recur
                        )
                        SubrogateRules.fit(method=ensemble,
                                           X_train=X_train,
                                           y_train= y_train,
                                           original_dataset=train_pandas_dataset,
                                           use_shap=False,
                                           use_lime=False)
                        y_pred_test_rules = SubrogateRules.predict(X_test, sorting_method=sorting)

                        if not y_pred_test_rules:
                            print("NOT CALCULATED")
                            continue

                        if len(y_pred_test_rules) == 0:
                            print("0 MATHS IN TEST")
                            empty_restuls(chi2, criteria, f, min_accuracy, min_class, scaler, y_test)
                            continue

                        line_results = generate_line_results(chi2, criteria, min_accuracy, min_class, SubrogateRules,
                                                             scaler,
                                                             sorting,
                                                             y_pred_test_RuleFit, y_pred_test_ensemble,
                                                             y_pred_test_rules,
                                                             y_pred_test_tree, y_test, y_pred_test_rulecosi)
                        print(line_results)
                        f.write(line_results)


def empty_restuls(chi2, criteria, f, min_accuracy, min_class, scaler, y_test):
    print('empty list no rules')
    tttt = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, NaN, {len(y_test)}, NaN, 0' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN\n'
    f.write(tttt)
    print(tttt)


def generate_line_results(chi2, criteria, min_accuracy, min_class, rules, scaler, sorting, y_pred_test_RuleFit,
                          y_pred_test_ensemble, y_pred_test_rules, y_pred_test_tree, y_test, y_pred_test_rulecosi):
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

    line_results = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(filtered_y_test)}, {len(rules.minimal_rules_)}, {len(filtered_y_pred_test_rules) / len(y_test)}'
    line_results += f', {ensemble_accuracy}, {ensemble_f1_score}, {ensemble_precision_score}, {ensemble_recall}, {ensemble_roc_auc_score}'
    line_results += f', {tree_accuracy}, {tree_f1_score}, {tree_precision_score}, {tree_recall_score}, {tree_roc_auc_score}'
    line_results += f', {RuleFit_accuracy}, {RuleFit_f1_score}, {RuleFit_precision_score}, {RuleFit_recall_score}, {RuleFit_roc_auc_score}'
    line_results += f', {rulecosi_accuracy}, {rulecosi_f1_score}, {rulecosi_precision_score}, {rulecosi_recall_score}, {rulecosi_roc_auc_score}'
    line_results += f', {rules_accuracy}, {rules_f1_score}, {rules_precision_score}, {rules_recall_score}, {rules_roc_auc_score}\n'
    return line_results


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


def generate_battery_test(recursive, classifier, f, filename, X, y, dataset, target_value_name, n_splits, n_repeats,
                          chi_square_percent_point_function,
                          scale_feature_coefficient, min_accuracy_coefficient, min_number_class_per_node,
                          sorting_method):
    cobertura_list, \
        RuleFit_accuracy_list, RuleFit_f1_score_list, RuleFit_precision_score_list, RuleFit_recall_list, RuleFit_roc_auc_score_list, \
        ensemble_accuracy_list, ensemble_f1_score_list, ensemble_precision_score_list, ensemble_recall_list, ensemble_roc_auc_score_list, \
        rules_accuracy_list, rules_f1_score_list, rules_precision_score_list, rules_recall_list, rules_roc_auc_score_list, \
        tree_accuracy_list, tree_f1_score_list, tree_precision_score_list, tree_recall_list, tree_roc_auc_score_list, \
        rulecosi_accuracy_list, rulecosi_f1_score_list, rulecosi_precision_score_list, rulecosi_recall_list, rulecosi_roc_auc_score_list, \
        rulefit_num_rules_list, rules_num_rules_list, rulecosi_num_rules_list, \
        ensemble_time_list, tree_time_list, RuleFit_time_list, rulecosi_time_list, rules_time_list \
        = kfold_test(recursive, classifier, X, chi_square_percent_point_function, dataset, min_accuracy_coefficient,
                     min_number_class_per_node, n_splits, n_repeats, scale_feature_coefficient, sorting_method,
                     target_value_name, y, filename)

    f_score = f'{filename} chi2:{chi_square_percent_point_function} minclass:{min_number_class_per_node}  min_precision:{min_accuracy_coefficient} scale_attr:{scale_feature_coefficient}'
    f_score += f',{classifier}'
    f_score += f',{recursive}'
    f_score += f',F1-score'
    f_score += f',{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    f_score += f',{round(tree_f1_score_list.mean() * 100, 2)}±{round(tree_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(ensemble_f1_score_list.mean() * 100, 2)}±{round(ensemble_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(RuleFit_f1_score_list.mean() * 100, 2)}±{round(RuleFit_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rulefit_num_rules_list.mean(), 2)}±{round(rulefit_num_rules_list.std(), 2)}'
    f_score += f',{round(rulecosi_f1_score_list.mean() * 100, 2)}±{round(rulecosi_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rulecosi_num_rules_list.mean(), 2)}±{round(rulecosi_num_rules_list.std(), 2)}'
    f_score += f',{round(rules_f1_score_list.mean() * 100, 2)}±{round(rules_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rules_num_rules_list.mean(), 2)}±{round(rules_num_rules_list.std(), 2)}\n'

    print(f_score)
    f.write(f_score)

    accuracy_score = f'{filename} chi2:{chi_square_percent_point_function} minclass:{min_number_class_per_node}  min_precision:{min_accuracy_coefficient} scale_attr:{scale_feature_coefficient}'
    accuracy_score += f',{classifier}'
    accuracy_score += f',{recursive}'
    accuracy_score += f',Accuracy-score'
    accuracy_score += f',{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    accuracy_score += f',{round(tree_accuracy_list.mean() * 100, 2)}±{round(tree_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(ensemble_accuracy_list.mean() * 100, 2)}±{round(ensemble_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(RuleFit_accuracy_list.mean() * 100, 2)}±{round(RuleFit_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rulefit_num_rules_list.mean(), 2)}±{round(rulefit_num_rules_list.std(), 2)}'
    accuracy_score += f',{round(rulecosi_accuracy_list.mean() * 100, 2)}±{round(rulecosi_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rulecosi_num_rules_list.mean(), 2)}±{round(rulecosi_num_rules_list.std(), 2)}'
    accuracy_score += f',{round(rules_accuracy_list.mean() * 100, 2)}±{round(rules_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rules_num_rules_list.mean(), 2)}±{round(rules_num_rules_list.std(), 2)}\n'

    print(accuracy_score)
    f.write(accuracy_score)

    precision_score = f'{filename} chi2:{chi_square_percent_point_function} minclass:{min_number_class_per_node}  min_precision:{min_accuracy_coefficient} scale_attr:{scale_feature_coefficient}'
    precision_score += f',{classifier}'
    precision_score += f',{recursive}'
    precision_score += f',Precision-Score'
    precision_score += f',{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    precision_score += f',{round(tree_precision_score_list.mean() * 100, 2)}±{round(tree_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(ensemble_precision_score_list.mean() * 100, 2)}±{round(ensemble_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(RuleFit_precision_score_list.mean() * 100, 2)}±{round(RuleFit_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rulefit_num_rules_list.mean(), 2)}±{round(rulefit_num_rules_list.std(), 2)}'
    precision_score += f',{round(rulecosi_precision_score_list.mean() * 100, 2)}±{round(rulecosi_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rulecosi_num_rules_list.mean(), 2)}±{round(rulecosi_num_rules_list.std(), 2)}'
    precision_score += f',{round(rules_precision_score_list.mean() * 100, 2)}±{round(rules_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rules_num_rules_list.mean(), 2)}±{round(rules_num_rules_list.std(), 2)}\n'

    print(precision_score)
    f.write(precision_score)

    recall = f'{filename} chi2:{chi_square_percent_point_function} minclass:{min_number_class_per_node}  min_precision:{min_accuracy_coefficient} scale_attr:{scale_feature_coefficient}'
    recall += f',{classifier}'
    recall += f',{recursive}'
    recall += f',Recall-Score'
    recall += f',{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    recall += f',{round(tree_recall_list.mean() * 100, 2)}±{round(tree_recall_list.std() * 100, 2)}'
    recall += f',{round(ensemble_recall_list.mean() * 100, 2)}±{round(ensemble_recall_list.std() * 100, 2)}'
    recall += f',{round(RuleFit_recall_list.mean() * 100, 2)}±{round(RuleFit_recall_list.std() * 100, 2)}'
    recall += f',{round(rulefit_num_rules_list.mean(), 2)}±{round(rulefit_num_rules_list.std(), 2)}'
    recall += f',{round(rulecosi_recall_list.mean() * 100, 2)}±{round(rulecosi_recall_list.std() * 100, 2)}'
    recall += f',{round(rulecosi_num_rules_list.mean(), 2)}±{round(rulecosi_num_rules_list.std(), 2)}'
    recall += f',{round(rules_recall_list.mean() * 100, 2)}±{round(rules_recall_list.std() * 100, 2)}'
    recall += f',{round(rules_num_rules_list.mean(), 2)}±{round(rules_num_rules_list.std(), 2)}\n'

    print(recall)
    f.write(recall)

    time = f'{filename} chi2:{chi_square_percent_point_function} minclass:{min_number_class_per_node}  min_precision:{min_accuracy_coefficient} scale_attr:{scale_feature_coefficient}'
    time += f',{classifier}'
    time += f',{recursive}'
    time += f',Time-Score'
    time += f',{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    time += f',{round(tree_time_list.mean() * 100, 2)}±{round(tree_time_list.std() * 100, 2)}'
    time += f',{round(ensemble_time_list.mean() * 100, 2)}±{round(ensemble_time_list.std() * 100, 2)}'
    time += f',{round(RuleFit_time_list.mean() * 100, 2)}±{round(RuleFit_time_list.std() * 100, 2)}'
    time += f',{round(rulefit_num_rules_list.mean(), 2)}±{round(rulefit_num_rules_list.std(), 2)}'
    time += f',{round(rulecosi_time_list.mean() * 100, 2)}±{round(rulecosi_time_list.std() * 100, 2)}'
    time += f',{round(rulecosi_num_rules_list.mean(), 2)}±{round(rulecosi_num_rules_list.std(), 2)}'
    time += f',{round(rules_time_list.mean() * 100, 2)}±{round(rules_time_list.std() * 100, 2)}'
    time += f',{round(rules_num_rules_list.mean(), 2)}±{round(rules_num_rules_list.std(), 2)}\n'

    print(time)
    f.write(time)

    return f_score, accuracy_score, precision_score, recall


def kfold_test(recursive, classifier, X, chi_square_percent_point_function, dataset, min_accuracy_coefficient,
               min_number_class_per_node, n_splits, n_repeats, scale_feature_coefficient, sorting_method,
               target_value_name, y, filename):
    cobertura_list = []
    rules_accuracy_list = []
    rules_f1_score_list = []
    rules_precision_score_list = []
    rules_recall_list = []
    rules_roc_auc_score_list = []
    ensemble_accuracy_list = []
    ensemble_f1_score_list = []
    ensemble_precision_score_list = []
    ensemble_recall_list = []
    ensemble_roc_auc_score_list = []
    tree_accuracy_list = []
    tree_f1_score_list = []
    tree_precision_score_list = []
    tree_recall_list = []
    tree_roc_auc_score_list = []
    RuleFit_accuracy_list = []
    RuleFit_f1_score_list = []
    RuleFit_precision_score_list = []
    RuleFit_recall_list = []
    RuleFit_roc_auc_score_list = []
    rulecosi_accuracy_list = []
    rulecosi_f1_score_list = []
    rulecosi_precision_score_list = []
    rulecosi_recall_list = []
    rulecosi_roc_auc_score_list = []
    # number of rules
    rulefit_num_rules_list = []
    rules_num_rules_list = []
    rulecosi_num_rules_list = []
    # time

    ensemble_time_list = []
    tree_time_list = []
    RuleFit_time_list = []
    rulecosi_time_list = []
    rules_time_list = []

    repeated_kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    for train, test in repeated_kfold.split(X, y):
        custom_scorer = make_scorer(f1_score, greater_is_better=True)
        param_grid_tree = {
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
        }

        if type(classifier) == type(RandomForestClassifier()) or type(LGBMClassifier()) or type(XGBClassifier()):
            param_grid = {
                'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
                'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
                'n_jobs': [2]  # threads
            }
        if type(classifier) == type(CatBoostClassifier()) or type(GradientBoostingClassifier()):
            param_grid = {
                'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
                'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
            }

        if type(classifier) == type(AdaBoostClassifier()):
            param_grid = {
                'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
            }
        # Random Forest
        clf_rf = GridSearchCV(
            # Evaluates the performance of different groups of parameters for a model based on cross-validation.
            classifier,
            param_grid,  # dict of parameters.
            cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
            scoring=custom_scorer)

        # TREE
        clf_tree = GridSearchCV(
            # Evaluates the performance of different groups of parameters for a model based on cross-validation.
            DecisionTreeClassifier(),
            param_grid_tree,  # dict of parameters.
            cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
            scoring=custom_scorer)

        X_train = X.loc[train].to_numpy()
        y_train = y.loc[train].to_numpy()
        X_test = X.loc[test].to_numpy()
        y_test = y.loc[test].to_numpy()

        X_train_int = X_train.astype(int)
        y_train_int = y_train.astype(int)
        X_test_int = X_test.astype(int)
        y_test_int = y_test.astype(int)

        train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                            columns=list(dataset['feature_names']) + [target_value_name])

        SubrogateRules = SRules(
            feature_names=dataset.feature_names,
            target_value_name=dataset.target_names,
            p_significance=chi_square_percent_point_function,
            scale_feature_coefficient=scale_feature_coefficient,
            min_accuracy_coefficient=min_accuracy_coefficient,
            minInsNode=min_number_class_per_node,
            display_features=False,
            display_logs=False,
            recursive=recursive
        )
        # Fit model
        start_time = time.time()
        clf_rf.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        ensemble_time_list.append(elapsed_time)

        ensemble = clf_rf.best_estimator_

        # RuleFit
        ruleFit = RuleFitClassifier()  # (tree_generator=ensemble)# alpha= [0.1, 1, 10, 100] include_linear= [True, False]

        # RULECOSI
        rulecosi = RuleCOSIClassifier(
            # base_ensemble=ensemble,
            conf_threshold=0.9,
            cov_threshold=0.0,
            column_names=dataset.feature_names)

        # Fit model
        rules_start_time = time.time()
        SubrogateRules.fit(ensemble, X_train, y_train, train_pandas_dataset, ensemble.feature_importances_)
        rules_time = time.time() - rules_start_time
        rules_time_list.append(rules_time)

        # TREE
        tree_start_time = time.time()
        clf_tree.fit(X_train, y_train)
        tree_elapsed_time = time.time() - tree_start_time
        tree_time_list.append(tree_elapsed_time)

        tree = clf_tree.best_estimator_

        # RULECOSI
        rulecosi_start_time = time.time()
        rulecosi.fit(X_train, y_train)
        rulecosi_elapsed_time = time.time() - rulecosi_start_time
        rulecosi_time_list.append(rulecosi_elapsed_time)

        RuleFit_start_time = time.time()
        ruleFit.fit(X_train_int, y_train_int, feature_names=dataset.feature_names)
        RuleFit_elapsed_time = time.time() - RuleFit_start_time
        RuleFit_time_list.append(RuleFit_elapsed_time)

        # Predict
        y_pred_test_ensemble = ensemble.predict(X_test)
        y_pred_test_rules = SubrogateRules.predict(X_test, sorting_method=sorting_method)
        y_pred_test_tree = tree.predict(X_test)

        y_pred_test_rulecosi = rulecosi.predict(X_test)

        y_pred_test_RuleFit = ruleFit.predict(X_test_int)

        # DATASET CATEGORIZABLES
        np_array_rules = np.array(y_pred_test_rules)
        filter_indices = np.where(np_array_rules != None)[0]
        filtered_y_test = np.array(y_test)[filter_indices].astype('int64')
        filtered_y_test_int = np.array(y_test_int)[filter_indices].astype('int64')

        if type(y_pred_test_ensemble[0]) == type("False"):
            y_pred_test_ensemble = np.array(y_pred_test_ensemble, dtype=bool)

        filtered_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype('int64')
        filtered_y_pred_test_tree = np.array(y_pred_test_tree)[filter_indices].astype('int64')
        filtered_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')

        filtered_y_pred_test_rulecosi = np.array(y_pred_test_rulecosi)[filter_indices].astype('int64')

        filtered_y_pred_test_RuleFit = np.array(y_pred_test_RuleFit)[filter_indices].astype('int64')

        if len(filter_indices) == 0:
            continue
        # SCORERS
        cobertura = len(filtered_y_pred_test_rules) / len(y_test)
        cobertura_list.append(cobertura)

        # Scores
        ensemble_accuracy, ensemble_f1_score, ensemble_precision_score, ensemble_recall, ensemble_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_ensemble)
        ensemble_accuracy_list.append(ensemble_accuracy)
        ensemble_f1_score_list.append(ensemble_f1_score)
        ensemble_precision_score_list.append(ensemble_precision_score)
        ensemble_recall_list.append(ensemble_recall)
        ensemble_roc_auc_score_list.append(ensemble_roc_auc_score)

        tree_accuracy, tree_f1_score, tree_precision_score, tree_recall, tree_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_tree)

        tree_accuracy_list.append(tree_accuracy)
        tree_f1_score_list.append(tree_f1_score)
        tree_precision_score_list.append(tree_precision_score)
        tree_recall_list.append(tree_recall)
        tree_roc_auc_score_list.append(tree_roc_auc_score)
        RuleFit_accuracy, RuleFit_f1_score, RuleFit_precision_score, RuleFit_recall, RuleFit_roc_auc_score = \
            generate_scores(filtered_y_test_int, filtered_y_pred_test_RuleFit)

        RuleFit_accuracy_list.append(RuleFit_accuracy)
        RuleFit_f1_score_list.append(RuleFit_f1_score)
        RuleFit_precision_score_list.append(RuleFit_precision_score)
        RuleFit_recall_list.append(RuleFit_recall)
        RuleFit_roc_auc_score_list.append(RuleFit_roc_auc_score)

        rules_accuracy, rules_f1_score, rules_precision_score, rules_recall, rules_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_rules)

        rules_accuracy_list.append(rules_accuracy)
        rules_f1_score_list.append(rules_f1_score)
        rules_precision_score_list.append(rules_precision_score)
        rules_recall_list.append(rules_recall)
        rules_roc_auc_score_list.append(rules_roc_auc_score)
        rulecosi_accuracy, rulecosi_f1_score, rulecosi_precision_score, rulecosi_recall, rulecosi_roc_auc_score = \
            generate_scores(filtered_y_test, filtered_y_pred_test_rulecosi)

        rulecosi_accuracy_list.append(rulecosi_accuracy)
        rulecosi_f1_score_list.append(rulecosi_f1_score)
        rulecosi_precision_score_list.append(rulecosi_precision_score)
        rulecosi_recall_list.append(rulecosi_recall)
        rulecosi_roc_auc_score_list.append(rulecosi_roc_auc_score)

        rules_num_rules_list.append(len(SubrogateRules.minimal_rules_))
        rulecosi_num_rules_list.append(len(rulecosi.simplified_ruleset_.rules))
        rulefit_num_rules_list.append(len(ruleFit.rules_))

    return np.array(cobertura_list), \
        np.array(RuleFit_accuracy_list), np.array(RuleFit_f1_score_list), np.array(
        RuleFit_precision_score_list), np.array(RuleFit_recall_list), np.array(RuleFit_roc_auc_score_list), \
        np.array(ensemble_accuracy_list), np.array(ensemble_f1_score_list), np.array(
        ensemble_precision_score_list), np.array(ensemble_recall_list), np.array(ensemble_roc_auc_score_list), \
        np.array(rules_accuracy_list), np.array(rules_f1_score_list), np.array(rules_precision_score_list), np.array(
        rules_recall_list), np.array(rules_roc_auc_score_list), \
        np.array(tree_accuracy_list), np.array(tree_f1_score_list), np.array(tree_precision_score_list), np.array(
        tree_recall_list), np.array(tree_roc_auc_score_list), \
        np.array(rulecosi_accuracy_list), np.array(rulecosi_f1_score_list), np.array(
        rulecosi_precision_score_list), np.array(rulecosi_recall_list), np.array(rulecosi_roc_auc_score_list), \
        np.array(rulefit_num_rules_list), np.array(rules_num_rules_list), np.array(rulecosi_num_rules_list), \
        np.array(ensemble_time_list), np.array(tree_time_list), np.array(RuleFit_time_list), np.array(
        rulecosi_time_list), np.array(rules_time_list)
