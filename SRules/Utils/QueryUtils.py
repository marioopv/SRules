import numpy as np


# TODO: TO NUMPY

@staticmethod
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


@staticmethod
def concatenate_query_comparer(full_feature_comparer):
    query = ''
    for g in full_feature_comparer:
        query = concatenate_query(query, g)
    return query


@staticmethod
def chunk_query(dataset_filtered, new_query):
    if not "&" in new_query:
        dataset_filtered = predict_unique_with_query(dataset_filtered, new_query)
    for group in divide_chunks(new_query.split("&"), 31):
        group_query = concatenate_query_comparer(group)
        dataset_filtered = predict_unique_with_query(dataset_filtered, group_query)
    return dataset_filtered


@staticmethod
def predict_unique_with_query(dataset, full_query):
    return dataset.query(full_query, engine='python')  # slower otherwise limit of 32 variables
    # return dataset.query(full_query)


@staticmethod
def concatenate_query(previous_full_query, rule_query):
    # A la query auxiliar se le incluye la característica del nodo junto con el valor asignado
    if previous_full_query != '':
        return f'{previous_full_query}  &  {rule_query}'
    return f'{rule_query}'


@staticmethod
def predict_unique_with_query_positives(dataset, feature_comparer, target_class_positive):
    dataset_filtered = dataset
    for comparer in feature_comparer:
        dataset_filtered = comparer.unitary_loc(dataset_filtered)
    dataset_filtered = target_class_positive.unitary_loc(dataset_filtered)
    return dataset_filtered


@staticmethod
def predict_unique_with_query_positives_query(dataset, full_feature_comparer, target_class_positive):
    dataset_filtered = dataset
    return predict_unique_with_query(dataset_filtered,
                                     concatenate_query(full_feature_comparer, target_class_positive.get_query()))


@staticmethod
def predict_unique_with_query_negatives(dataset, feature_comparer, target_class_negative):
    dataset_filtered = dataset
    for comparer in feature_comparer:
        dataset_filtered = comparer.unitary_loc(dataset_filtered)
    dataset_filtered = target_class_negative.unitary_loc(dataset_filtered)
    return dataset_filtered


@staticmethod
def predict_unique_with_query_negatives_query(dataset, full_feature_comparer, target_class_negative):
    dataset_filtered = dataset
    return predict_unique_with_query(dataset_filtered, concatenate_query(full_feature_comparer,
                                                                         target_class_negative.get_query()))


@staticmethod
def count_query_positives(dataset, feature_comparer, target_class_positive):
    return len(predict_unique_with_query_positives(dataset, feature_comparer, target_class_positive))


@staticmethod
def count_query_negatives(dataset, feature_comparer, target_class_negative):
    return len(predict_unique_with_query_negatives(dataset, feature_comparer, target_class_negative))


@staticmethod
def count_query_positives_query(dataset, full_query, target_class_positive):
    return len(predict_unique_with_query_positives_query(dataset, full_query, target_class_positive))


@staticmethod
def count_query_negatives_query(dataset, full_query, target_class_negative):
    return len(predict_unique_with_query_negatives_query(dataset, full_query, target_class_negative))
