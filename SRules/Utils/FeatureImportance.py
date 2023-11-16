
from sklearn import preprocessing
import numpy as np
from sklearn.svm import LinearSVC
from SRules.Utils import DisplayUtils
import SRules.Utils.LimeUtils as LimeUtils
import SRules.Utils.ShapUtils as ShapUtils
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

@staticmethod
def extract_feature_importances( method, X_train, use_shap, use_lime):
    if type(method) == type(RandomForestClassifier())\
            or type(method) == type(MLPClassifier())\
            or type(method) == type(LinearSVC())\
            or type(method) == type(LogisticRegression()):
        if use_shap:
            return ShapUtils.extract_feature_importances_shap(method, X_train)
        if use_lime:
            return LimeUtils.extract_feature_importances_lime(method)

#TODO: CHECK OTHER METHODS

    return method.feature_importances_
@staticmethod
def normalized_features(feature_importance_list):
    return preprocessing.MinMaxScaler().fit_transform(feature_importance_list.reshape(-1, 1))


@staticmethod
def get_top_important_features_list(feature_importances, feature_names, scale_feature_coefficient, display_logs, display_features):
    """
    Obtiene las características más importantes en orden descendente
    :return:
    :param coefficient: Coeficiente entre 0 y 1 usado para obtener un % de las características más importantes.
    :param feature_names: Lista de los nombres de las columnas del dataset.
    :param feature_importances: Valor de importancia asociado a cada característica en el modelo entrenado.
    :return: Ordered feature list
    """

    if display_logs:
        print("->Extract feature importance list")

    # Feature Importance list

    # Índices de las características más significativas ordenadas
    index = np.argsort(feature_importances)[::-1].tolist()

    X_train_minmax = normalized_features(feature_importances)
    if display_features:
        DisplayUtils.plot_features(X_train_minmax)

    most_important_features_ = [feature_names[x] for x in index if
                                     X_train_minmax[x] >= scale_feature_coefficient]

    if display_logs:
        print(f'\t Original features {len(feature_importances)}')
        print(f'\t Selected features {len(most_important_features_)}')
        print(
            f'\t Percentage of selected rules: {100 * len(most_important_features_) / len(feature_importances)} %')

    return most_important_features_, feature_importances