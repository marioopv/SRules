class FeatureComparer:
    def __init__(self, feature_name, comparer, value):
        self.feature_name = feature_name
        self.comparer = comparer
        self.value = value

    def __str__(self):
        return self.get_query()

    def get_query(self):
        return f'{self.feature_name} {self.comparer} {str(self.value)}'

    def unitary_loc(self, dataset):
        return dataset.loc[dataset[self.feature_name] == self.value]