import numpy as np
from SRules.Utils.QueryUtils import concatenate_query


class Pattern:
    def __init__(self,
                 target_value,
                 feature_names,
                 full_feature_comparer,
                 chi2_statistic,
                 p_value,
                 chi2_critical_value,
                 expected_freq,
                 number_target,
                 number_all,
                 target_accuracy):
        self.target_value = target_value  # str
        self.p_value = p_value  # str
        self.chi2_statistic = chi2_statistic  # str
        self.chi2_critical_value = chi2_critical_value  # str
        self.expected_freq = expected_freq  # str
        self.full_feature_comparer = full_feature_comparer  # Node
        self.number_target = number_target
        self.number_all = number_all
        self.target_accuracy = target_accuracy
        self.feature_names = feature_names

    def get_complexity(self):
        return len(self.full_feature_comparer)

    def get_full_rule(self):
        full_query = ''
        for comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, f'{comparer.get_query()}')
        return full_query

    def Predict(self, data_array):
        for comparer in self.full_feature_comparer:
            index = np.where(self.feature_names == comparer.feature_name)[0][0]
            if float(comparer.value) != float(data_array[index]):
                return None
        return self.target_value

    def __str__(self):
        display = '> ------------------------------\n'
        display += f' ** Target value: {self.target_value}'
        display += f' ** Target: {self.number_target}'
        display += f' ** Total: {self.number_all}'
        display += f' ** Accuracy: {self.target_accuracy}'
        display += f' ** Complexity: {self.get_complexity()}'
        display += f' ** Chi2 critical_value: {self.chi2_critical_value}'
        display += f' ** P_value: {self.p_value}\n'
        display += f'\t Query: {self.get_full_rule()}\n'
        display += '> ------------------------------\n'
        return display