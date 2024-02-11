import numpy as np
from scipy.stats import chi2_contingency, chi2
@staticmethod
def parent_relation_matrix(nodes_dict, children):
    # (Recordar que solo busco nodos padre para calcular el chi-square del total de sus hijos)
    aux_matrix = []
    for child_id in children:  # Access every single sibling node
        aux_node = nodes_dict[child_id]
        aux_matrix.append([aux_node.number_positives, aux_node.number_negatives])
    # Se calcula el p valor de los hermanos en ese subnivel
    return np.array(aux_matrix).astype(float).transpose()


@staticmethod
# Calcula los coeficientes de chi-square usando los valores de muertes y
# supervivencias del nodo en cuestión y del nodo padre
def chi2_values(nodes_dict, children, chi_square_percent_point_function):
    # https://matteocourthoud.github.io/post/chisquared/
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    # https://towardsdatascience.com/gentle-introduction-to-chi-square-test-for-independence-7182a7414a95#92ef

    matrix = parent_relation_matrix(nodes_dict, children)
    matrix[matrix == 0] = 0.0001
    chi2_results = chi2_contingency(matrix, correction=False)
    chi2_critical_value = chi2.ppf(chi_square_percent_point_function, chi2_results.dof)

    return chi2_results.statistic, chi2_results.pvalue, chi2_results.dof, chi2_results.expected_freq, chi2_critical_value
