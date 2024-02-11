import copy
import time
import SRules.Utils.QueryUtils as q
from SRules.Pattern import Pattern
from SRules.Utils import Chi2Utils


@staticmethod
def join_all_rules(all_rules_):
    rules = []
    for rule_list in all_rules_:
        for rule in rule_list:
            rules.append(rule)
    return rules


@staticmethod
def prune_rules(minimal_rules, sorted_rules, display_logs):
    start_time = time.time()
    if display_logs:
        print("->Prune Rules")

    if minimal_rules is None:
        minimal_rules = []

    for candidate_rule in sorted_rules:
        candidate_full_rule = candidate_rule.get_full_rule()
        should_include = True

        for minimal_rule in minimal_rules:  # TODO: CHECKING TWICE??
            minimal_full_rule = minimal_rule.get_full_rule()
            if minimal_full_rule in candidate_full_rule:
                # no valida
                should_include = False
                break
        if should_include:
            minimal_rules.append(candidate_rule)

    elapsed_time = time.time() - start_time
    if display_logs:
        print(f"Elapsed time to compute the prune_rules: {elapsed_time:.3f} seconds")

    return minimal_rules


@staticmethod
def obtain_pattern_list_of_valid_nodes_with_p_value(nodes_dict, chi_square_percent_point_function, feature_names,
                                                    display_logs):
    """
    Construct the list of rules based on the chi square of their sons
    @return: pattern_list_valid_nodes
    """

    start_time = time.time()
    if display_logs:
        print("->Generate obtained patterns tree")
    pattern_list_valid_nodes = []
    visited_nodes = []  # Lista auxiliar para guardar los IDs de los nodos que ya han sido visitados.
    # Visita todos los nodos, y de aquellos que no sean el nodo principal y que tengan hijos, obtiene el chi-square de los hijos de ese nodo.
    for key, node in nodes_dict.items():
        if node.children is None:
            continue
        if node.node_id in visited_nodes:
            continue
        visited_nodes.append(node.node_id)
        # Obtiene la lista de IDs de sus nodos hermanos
        children = node.children
        # En el caso de que ese nodo no sea un nodo hoja
        if len(children) > 0:
            chi2_statistic, p_value, degrees_of_freedom, expected_freq, chi2_critical_value = Chi2Utils.chi2_values(nodes_dict,
                                                                                                          children,
                                                                                                          chi_square_percent_point_function)

            if chi2_statistic > chi2_critical_value:
                for child_id in children:
                    child = nodes_dict[child_id]
                    # Set rules and last value to NONE
                    current_full_feature_comparer = copy.deepcopy(child.full_feature_comparer)
                    current_full_feature_comparer[-1].value = None

                    # Si se encuentra una regla que puede tener un patrón, se incluye.
                    pattern = Pattern(target_value=None,
                                      full_feature_comparer=current_full_feature_comparer,
                                      p_value=p_value,
                                      chi2_statistic=chi2_statistic,
                                      chi2_critical_value=chi2_critical_value,
                                      expected_freq=expected_freq,
                                      number_target=None,  # define later
                                      feature_names=feature_names,
                                      number_all=None,
                                      target_accuracy=None)
                    pattern_list_valid_nodes.append(pattern)

    elapsed_time = time.time() - start_time
    if display_logs:
        print(
            f"Elapsed time to compute the obtain_pattern_list_of_valid_nodes_with_pvalue: {elapsed_time:.3f} seconds")
    return pattern_list_valid_nodes



@staticmethod
def categorize_patterns(test_data,
                        pattern_list_valid_nodes,
                        min_accuracy_coefficient,
                        target_class_positive,
                        target_class_negative,
                        target_true,
                        target_false,
                        display_logs):
    """
    PSEUDO FIT
    :param test_data:
    :param pattern_list_valid_nodes:
    :return: list of rules
    """
    rules_ = []
    start_time = time.time()
    if display_logs:
        print("->Categorize patterns")

    # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
    index = 0
    # TODO: PARALLEL
    while index < len(pattern_list_valid_nodes):
        for distinct_value in [target_true, target_false]:
            # UPDATE VALUES
            new_rule = copy.deepcopy(pattern_list_valid_nodes[index])
            new_rule.full_feature_comparer[-1].value = distinct_value

            # Get values for positives and negatives
            number_negatives = q.count_query_negatives_query(test_data, new_rule.get_full_rule(), target_class_negative)
            number_positives = q.count_query_positives_query(test_data, new_rule.get_full_rule(), target_class_positive)
            number_positives_and_negatives = number_positives + number_negatives

            # If this rule has existing cases in total in the training set, is included.
            if number_positives_and_negatives > 0:
                new_rule.number_all = number_positives_and_negatives
                # Checks if the combinations show a rule for negative/positives
                proportion_positives = number_positives / number_positives_and_negatives

                # do not include rules with 0.5 prob
                if proportion_positives == 0.5:
                    continue

                if proportion_positives >= min_accuracy_coefficient:
                    # POSITIVES
                    new_rule.target_value = target_true
                    new_rule.number_target = number_positives
                    new_rule.target_accuracy = proportion_positives
                else:
                    # NEGATIVES
                    proportion_negatives = number_negatives / number_positives_and_negatives
                    if proportion_negatives >= min_accuracy_coefficient:
                        new_rule.target_value = target_false
                        new_rule.number_target = number_negatives
                        new_rule.target_accuracy = proportion_negatives
                    else:
                        continue
                rules_.append(new_rule)
        index += 1

    elapsed_time = time.time() - start_time
    if display_logs:
        print(f"Elapsed time to compute the categorize_patterns: {elapsed_time:.3f} seconds")

    return rules_

@staticmethod
def rules_description(rules_, minimal_rules_):
    display = '> ++++++++++++++++++++++++++++\n'
    if rules_ is not None:
        display += f'> SRules --  Number of Rules: {len(rules_)}\n'
    if minimal_rules_ is not None:
        display += f'> SRules --  Number of Minimal Rules: {len(minimal_rules_)}\n'
    display += '> ++++++++++++++++++++++++++++\n'
    return display


@staticmethod
def print_rules(rules_):
    display = '> ------COMPLETE RULES--------\n'
    for num in range(len(rules_)):
        display += f'{rules_[num]}'
    return display


@staticmethod
def print_minimal_rules(minimal_rules_):
    display = '> ------MINIMAL RULES--------\n'
    if minimal_rules_ is not None:
        for num in range(len(minimal_rules_)):
            display += f'{minimal_rules_[num]}'
    return display
