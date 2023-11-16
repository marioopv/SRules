import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

@staticmethod
def extract_feature_importances_shap(method, X_train):

    if type(method) == type(LogisticRegression()):
        explainer = shap.Explainer(method, X_train)
    else:
        explainer = shap.TreeExplainer(method)
    shap_values = explainer.shap_values(X_train)
    np_shap_values = np.array(shap_values)

    if type(method) == type(LogisticRegression()):
        shape = 1
    else:
        shape = np_shap_values.shape[0]

    for shap_val in range(shape):
        np_shap_values = np_shap_values.mean(axis=0)

    return np_shap_values
