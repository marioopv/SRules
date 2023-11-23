import shap
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from SRules.Utils import DisplayUtils
import warnings

warnings.filterwarnings("ignore")


@staticmethod
def extract_feature_importance_shap(method, X_train):
    print(f" ----> SHAP - EXPLAINER")

    # LINEAR MODEL
    if type(method) == type(LogisticRegression()):
        explainer = shap.Explainer(method, X_train)

    # MODEL SPECIFIC
    if type(method) == type(RandomForestClassifier()) \
            or type(method) == type(GradientBoostingClassifier()) \
            or type(method) == type(AdaBoostClassifier()) \
            or type(method) == type(LGBMClassifier()) \
            or type(method) == type(XGBClassifier()) \
            or type(method) == type(CatBoostClassifier()):
        explainer = shap.TreeExplainer(method)

    # MODEL NEURAL NETWORK
    # DeepExplainer
    # https://github.com/shap/shap/blob/master/notebooks/genomic_examples/DeepExplainer%20Genomics%20Example.ipynb
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html
    # GradientExplainer


    # MODEL AGNOSTIC
    if type(method) == type(LinearSVC()) \
            or type(method) == type(SVC()) \
            or type(method) == type(KNeighborsClassifier()) \
            or type(method) == type(MLPClassifier()):
        explainer = shap.KernelExplainer(method.predict, X_train)  # , keep_index=True


    print(f" ----> SHAP - VALUES")
    np_shap_values = np.array(explainer.shap_values(X_train))

    # print(np_shap_values)

    if type(method) == type(LogisticRegression()):
        shape = 1
    else:
        shape = np_shap_values.shape[0]

    np_shap_values = np.absolute(np_shap_values)

    for shap_val in range(shape):
        np_shap_values = np_shap_values.mean(axis=0)

    DisplayUtils.plot_features(np_shap_values)
    return np_shap_values
