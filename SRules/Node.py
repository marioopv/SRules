from SRules.Utils.QueryUtils import concatenate_query


class Node:
    def __init__(self,
                 node_id,
                 parent_id,
                 number_negatives,
                 number_positives,
                 full_feature_comparer = []):
        self.node_id = node_id  # int
        self.parent_id = parent_id  # int
        self.number_negatives = number_negatives  # int
        self.number_positives = number_positives  # int
        self.full_feature_comparer = full_feature_comparer
        self.children = []  # list of int. Contiene todos los IDs de los hijos

    def get_full_query(self):
        full_query = ''
        for feature_comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, feature_comparer.get_query())
        return full_query

    def __str__(self):
        display = '> ------------------------------\n'
        display += f'>** Node ID: {self.node_id} '
        display += f'** Numer of comparisons: {len(self.full_feature_comparer)}:\n'
        display += '> ------------------------------\n'
        for feature_comparer in self.full_feature_comparer:
            display += f'\t{feature_comparer}\n'
        return display
