import time
import copy
import numpy as np
from scipy.stats import chi2_contingency, chi2
from sklearn.base import ClassifierMixin

from SRules.FeatureComparer import FeatureComparer
from SRules.Node import Node
from SRules.Pattern import Pattern
import SRules.Utils.QueryUtils as q
import SRules.Utils.FeatureImportance as FeatureImportance
from SRules.Utils import RulesUtils, DatasetUtils


# TODO: MOVE STATIC METHODS TO OTHER FILES
# FETURE IMPORTANCE

class SRules(ClassifierMixin):
    display_logs: bool
    display_features: bool
    target_true: bool
    target_false: bool
    recursive: bool

    def __init__(self,
                 feature_names,
                 target_value_name="target",
                 display_logs=False,
                 display_features=False,
                 target_true=True,
                 target_false=False,
                 recursive=False,
                 chi_square_percent_point_function=0.95,
                 scale_feature_coefficient=0.2,
                 min_accuracy_coefficient=0.9,
                 min_number_class_per_node=3
                 ):
        self.rules_ = []
        self.minimal_rules_ = []
        self.feature_importance = None
        self.most_important_features_ = None
        self.nodes_dict = {}
        self.nodes_dict_ids = []
        self.pattern_list_valid_nodes = []
        self.feature_names = feature_names
        self.target_value_name = target_value_name
        self.target_true = target_true
        self.target_false = target_false
        self.target_class_positive = FeatureComparer(target_value_name, '==', self.target_true)
        self.target_class_negative = FeatureComparer(target_value_name, '==', self.target_false)
        self.chi_square_percent_point_function = chi_square_percent_point_function
        self.scale_feature_coefficient = scale_feature_coefficient
        self.min_accuracy_coefficient = min_accuracy_coefficient
        self.min_number_class_per_node = min_number_class_per_node
        self.display_features = display_features
        self.display_logs = display_logs
        self.recursive = recursive
        self.all_rules_ = []

    def get_node(self, node_id):
        return self.nodes_dict[node_id]

    def parent_relation_matrix(self, children):
        # (Recordar que solo busco nodos padre para calcular el chi-square del total de sus hijos)
        aux_matrix = []
        for child_id in children:  # Access every single sibling node
            aux_node = self.get_node(child_id)
            aux_matrix.append([aux_node.number_positives, aux_node.number_negatives])
        # Se calcula el p valor de los hermanos en ese subnivel
        return np.array(aux_matrix).astype(float).transpose()

    # Calcula los coeficientes de chi-square usando los valores de muertes y
    # supervivencias del nodo en cuestión y del nodo padre
    def chi2_values(self, children):
        # https://matteocourthoud.github.io/post/chisquared/
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        # https://towardsdatascience.com/gentle-introduction-to-chi-square-test-for-independence-7182a7414a95#92ef

        matrix = self.parent_relation_matrix(children)
        matrix[matrix == 0] = 0.0001
        chi2_results = chi2_contingency(matrix, correction=False)
        chi2_critical_value = chi2.ppf(self.chi_square_percent_point_function, chi2_results.dof)

        return chi2_results.statistic, chi2_results.pvalue, chi2_results.dof, chi2_results.expected_freq, chi2_critical_value

    def obtain_pattern_list_of_valid_nodes_with_p_value(self):
        """
        Construct the list of rules based on the chi square of their sons
        @return: pattern_list_valid_nodes
        """

        start_time = time.time()
        if self.display_logs:
            print("->Generate obtained patterns tree")

        visited_nodes = []  # Lista auxiliar para guardar los IDs de los nodos que ya han sido visitados.
        # Visita todos los nodos, y de aquellos que no sean el nodo principal y que tengan hijos, obtiene el chi-square de los hijos de ese nodo.
        for key, node in self.nodes_dict.items():
            if node.children is None:
                continue
            if node.node_id in visited_nodes:
                continue
            visited_nodes.append(node.node_id)
            # Obtiene la lista de IDs de sus nodos hermanos
            children = node.children
            # En el caso de que ese nodo no sea un nodo hoja
            if len(children) > 0:
                chi2_statistic, p_value, degrees_of_freedom, expected_freq, chi2_critical_value = self.chi2_values(
                    children)

                if chi2_statistic > chi2_critical_value:
                    for child_id in children:
                        child = self.get_node(child_id)
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
                                          feature_names=self.feature_names,
                                          number_all=None,
                                          target_accuracy=None)
                        self.pattern_list_valid_nodes.append(pattern)

        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(
                f"Elapsed time to compute the obtain_pattern_list_of_valid_nodes_with_pvalue: {elapsed_time:.3f} seconds")

    def categorize_patterns(self, test_data):
        """
        PSEUDO FIT
        :param test_data:
        :param pattern_list_valid_nodes:
        :return: list of rules
        """

        start_time = time.time()
        if self.display_logs:
            print("->Categorize patterns")

        # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
        index = 0
        # TODO: PARALLEL
        while index < len(self.pattern_list_valid_nodes):
            for distinct_value in [self.target_true, self.target_false]:
                # UPDATE VALUES
                new_rule = copy.deepcopy(self.pattern_list_valid_nodes[index])
                new_rule.full_feature_comparer[-1].value = distinct_value

                # Get values for positives and negatives
                number_negatives = q.count_query_negatives_query(test_data, new_rule.get_full_rule(),
                                                                 self.target_class_negative)
                number_positives = q.count_query_positives_query(test_data, new_rule.get_full_rule(),
                                                                 self.target_class_positive)
                number_positives_and_negatives = number_positives + number_negatives

                # If this rule has existing cases in total in the training set, is included.
                if number_positives_and_negatives > 0:
                    new_rule.number_all = number_positives_and_negatives
                    # Checks if the combinations show a rule for negative/positives
                    proportion_positives = number_positives / number_positives_and_negatives

                    # do not include rules with 0.5 prob
                    if proportion_positives == 0.5:
                        continue

                    if proportion_positives >= self.min_accuracy_coefficient:
                        # POSITIVES
                        new_rule.target_value = self.target_true
                        new_rule.number_target = number_positives
                        new_rule.target_accuracy = proportion_positives
                    else:
                        # NEGATIVES
                        proportion_negatives = number_negatives / number_positives_and_negatives
                        if proportion_negatives >= self.min_accuracy_coefficient:
                            new_rule.target_value = self.target_false
                            new_rule.number_target = number_negatives
                            new_rule.target_accuracy = proportion_negatives
                        else:
                            continue
                    self.rules_.append(new_rule)
            index += 1

        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(f"Elapsed time to compute the categorize_patterns: {elapsed_time:.3f} seconds")

        return self.rules_

    def binary_tree_generator(self,
                              dataset,
                              node_value=0,
                              feature_index=0,
                              parent_node=None):
        """
        Función recursiva encargada de generar el árbol de nodos con sus respectivas queries y obtener en cada nodo la query y el número de fallecimientos y supervivencias de cada uno.

        :param dataset: DataFrame. Dataset con las filas para obtener el número de fallecimientos y defunciones usando cada query.
        :param node_value: Representa el valor de la característica en ese nodo en concreto.
        :param feature_index: índice auxiliar de la lista de características
        :param parent_node: node of the parent of current node
        :return:
        """
        if feature_index >= len(self.most_important_features_):
            # No hay más niveles
            return

        current_feature_name = self.most_important_features_[feature_index]

        feature_comparer = FeatureComparer(current_feature_name, '==', node_value)
        # Caso base para el que se considera como nodo padre de todos.
        if parent_node is None:
            # Create Node
            current_node = Node(node_id=0,
                                parent_id=None,
                                number_positives=q.count_query_positives_query(dataset, '', self.target_class_positive),
                                number_negatives=q.count_query_negatives_query(dataset, '', self.target_class_negative),
                                full_feature_comparer=[])
            # Incluye el nodo en la lista
            self.nodes_dict[current_node.node_id] = current_node
            self.nodes_dict_ids.append(current_node.node_id)

            # Una vez creado el padre, se accede a la primera característica, que representaría el primer nivel.
            # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
            for node_value in dataset[current_feature_name].unique():
                self.binary_tree_generator(dataset, node_value=node_value, parent_node=current_node)

        # Caso en el que el padre ya ha sido creado
        else:
            full_rule_query = q.concatenate_query(parent_node.get_full_query(), feature_comparer.get_query())
            number_negatives = q.count_query_negatives_query(dataset, full_rule_query, self.target_class_negative)
            number_positives = q.count_query_positives_query(dataset, full_rule_query, self.target_class_positive)

            full_comparer = parent_node.full_feature_comparer + [feature_comparer]

            node_values_total = number_negatives + number_positives
            # Si el nodo se considera que no tiene los casos suficientes,
            # es descartado y el árbol no continúa en esa rama.
            if node_values_total >= self.min_number_class_per_node:
                # Se le asigna la ID al nodo como la siguiente a la última utilizada.

                node_dict_ID = self.nodes_dict_ids[-1] + 1
                current_node = Node(node_id=node_dict_ID,
                                    parent_id=parent_node.node_id,
                                    number_negatives=number_negatives,
                                    number_positives=number_positives,
                                    full_feature_comparer=full_comparer
                                    )

                # Incluye el nodo en la lista
                self.nodes_dict[current_node.node_id] = current_node
                self.nodes_dict_ids.append(current_node.node_id)

                # La ID del nodo es incluida en la lista de hijos del padre.
                self.nodes_dict[parent_node.node_id].children.append(node_dict_ID)

                # new_dataset = dataset.loc[:, dataset.columns != current_feature_name]
                new_feature_index = feature_index + 1
                if new_feature_index >= len(self.most_important_features_):
                    # No hay más niveles
                    return
                # Por cada posible valor que pueda tomar esa nueva característica, se crea un hijo nodo de manera recursiva
                for node_value in dataset[self.most_important_features_[new_feature_index]].unique():
                    self.binary_tree_generator(dataset, node_value=node_value,
                                               feature_index=new_feature_index,
                                               parent_node=current_node)

    def generate_nodes(self, dataset, feature_importances):
        # List of top % important features in the model are obtained. This % regulated by coefficient between [0,1].
        if self.most_important_features_ is None or []:
            self.most_important_features_, self.feature_importance = FeatureImportance.get_top_important_features_list(
                feature_importances,
                self.feature_names,
                self.scale_feature_coefficient,
                self.display_logs,
                self.display_features)

        if self.most_important_features_ is None or []:
            return None, None, None

        # Generate Tree

        _, minimal_dataset = self.generate_tree(dataset=dataset)
        return self.nodes_dict, minimal_dataset, self.most_important_features_

    def generate_tree(self, dataset):
        # Genera el árbol binario y obtiene las combinaciones que indican que hay un patrón:

        # TODO: GENERATE METHOD TO minimal_dataset
        if self.most_important_features_ is [] or None or len(self.most_important_features_) == 0:
            return False, dataset

        minimal_dataset = copy.deepcopy(dataset[self.define_minimal_columns()])
        minimal_dataset.sort_values(self.most_important_features_, inplace=True, ascending=True)

        if minimal_dataset is [] or None or len(minimal_dataset) == 0:
            return False, dataset

        if self.display_logs:
            print("->Generate new tree based on list")
        start_time = time.time()
        self.binary_tree_generator(dataset=minimal_dataset)
        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(f"Elapsed time to compute the binary_tree_generator: {elapsed_time:.3f} seconds")

        return self.nodes_dict, minimal_dataset

    def define_minimal_columns(self):
        return self.most_important_features_ + [self.target_value_name]

    def single_fit(self, dataset,
                   feature_importance,
                   node_dict=None,
                   sorting_method="target_accuracy"):
        """
        Get list of top features and generate rules
        :param dataset:
        :return:
        @type dataset: dataset
        @type node_dict: object
        @param node_dic
        @param feature_importance:
        """

        minimal_dataset = None
        if node_dict is not None:
            self.nodes_dict = node_dict

        # if dict is null calculate it
        if not self.nodes_dict:
            _, minimal_dataset, _1 = self.generate_nodes(dataset, feature_importance)
            if minimal_dataset is None and _ is None and _1 is None:
                return

        if self.most_important_features_ is None or []:
            self.most_important_features_, self.feature_importance = FeatureImportance.get_top_important_features_list(
                feature_importance,
                self.feature_names,
                self.scale_feature_coefficient,
                self.display_logs,
                self.display_features)

        if self.most_important_features_ is None or []:
            return

        # Lista de nodos válidos
        self.obtain_pattern_list_of_valid_nodes_with_p_value()

        # TODO: GENERATE METHOD TO minimal_dataset
        if self.most_important_features_ is not None or []:
            minimal_dataset = copy.deepcopy(dataset[self.define_minimal_columns()])
            minimal_dataset.sort_values(self.most_important_features_, inplace=True, ascending=True)

        if minimal_dataset is None or []:
            return

        # Categoriza patrones
        self.categorize_patterns(minimal_dataset)

        # Sort rules AND Prune rules
        self.minimal_rules_ = RulesUtils.prune_rules(self.minimal_rules_,
                                                     self.sorting(sorting_method),
                                                     self.display_logs)

        self.all_rules_.append(self.minimal_rules_)

        return self

    def fit(self,
            method,
            X_train,
            y_train,
            original_dataset,
            use_shap=False,
            use_lime=False,
            sorting_method="target_accuracy"):
        """
        Get list of top features and generate rules
        Parameters
        ----------
        original_dataset
        method
        X_train : object
        y_train
        """
        print("INIT")

        dataset = copy.deepcopy(original_dataset)
        X_train = copy.deepcopy(X_train)
        y_train = copy.deepcopy(y_train)

        self.feature_importance = FeatureImportance.extract_feature_importance(method, X_train, use_shap, use_lime)

        print(" -> TRAINING MODEL")

        if self.recursive is False:
            print(" --> SINGLE FIT")
            return self.single_fit(dataset=dataset,
                            feature_importance=self.feature_importance,
                            sorting_method="target_accuracy")

        print(" --> RECURSIVE FIT")
        return self.recursive_fit(X_train=X_train,
                                  dataset=dataset,
                                  method=method,
                                  sorting_method=sorting_method,
                                  use_lime=use_lime,
                                  use_shap=use_shap,
                                  y_train=y_train)

    def recursive_fit(self,
                      X_train,
                      dataset,
                      method,
                      sorting_method,
                      use_lime,
                      use_shap,
                      y_train):

        recursive_counter = 1
        previous_dataset_len = len(X_train)
        feature_importance = self.feature_importance

        print(" ---> Fitting Recursive Model")
        while True:
            print(f" ----> Step: {recursive_counter}")
            self.single_fit(dataset,
                            feature_importance=feature_importance,
                            sorting_method="target_accuracy")

            print(f" ----> {recursive_counter} - Model Prediction")
            # predict
            y_pred_train_rules = self.predict(X_train, sorting_method)

            # New datasets
            print(f" ----> {recursive_counter} - Creating new dataset")
            X_train, y_train, dataset, new_len = (DatasetUtils
                                                  .new_datasets(X_train=X_train,
                                                                y_train=y_train,
                                                                y_pred_train_rules=y_pred_train_rules,
                                                                dataset=dataset))

            print(f' ----> {recursive_counter} - Previous dataset length: {previous_dataset_len}')
            print(f' ----> {recursive_counter} - New dataset length: {new_len}')

            if self.display_logs:
                print(self.rules_description())

            # No improvement or
            if new_len == 0 or previous_dataset_len == new_len:
                print(" ---> Finishing fitting model")
                break
            previous_dataset_len = new_len

            print(f" ----> {recursive_counter} - Fitting method")
            method.fit(X_train, y_train)

            print(f" ----> {recursive_counter} - Feature importance")
            feature_importance = FeatureImportance.extract_feature_importance(method, X_train, use_shap, use_lime)

            # Clean variables
            self.clean()
            recursive_counter += 1

        # Join Rules
        print(" ---> Joining all rules")
        self.rules_ = RulesUtils.join_all_rules(self.all_rules_)
        # Prune rules
        print(" ---> Pruning rules")
        self.minimal_rules_ = RulesUtils.prune_rules(None,
                                                     self.sorting(),
                                                     self.display_logs)
        print("END")
        return self

    # TODO: Extract TO STATIC?

    def clean(self):
        self.rules_ = []
        self.minimal_rules_ = []
        self.feature_importance = None
        self.nodes_dict = {}
        self.nodes_dict_ids = []
        self.pattern_list_valid_nodes = []
        self.most_important_features_ = None

    def predict(self, X, sorting_method="target_accuracy"):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with the highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
            @param X:
            @param sorting_method:
            @return:
        """
        predictions = []

        if self.minimal_rules_ is None:
            print("****************************")
            print("PRUNNING RULES IN PREDICTION")
            print("****************************")
            self.minimal_rules_ = RulesUtils.prune_rules(self.minimal_rules_,
                                                         self.sorting(sorting_method),
                                                         self.display_logs)

        for x in X:
            categorize_rule = None
            for rule in self.minimal_rules_:
                prediction = rule.Predict(x)
                if prediction is not None:
                    predictions.append(prediction)
                    categorize_rule = True
                    break
            if categorize_rule is None:
                predictions.append(None)

        return predictions

    def sorting(self, sorting_method="target_accuracy"):
        match sorting_method:
            case "target_accuracy":
                return sorted(self.rules_, key=lambda r: r.target_accuracy, reverse=True)
            case "complexity":
                return sorted(self.rules_, key=lambda r: r.get_complexity(), reverse=True)
            case "p_value":
                return sorted(self.rules_, key=lambda r: r.p_value)
            case "chi2_statistic":
                return sorted(self.rules_, key=lambda r: r.chi2_statistic, reverse=True)
            case default:
                return sorted(self.rules_, key=lambda r: r.target_accuracy, reverse=True)

    def rules_description(self):
        display = '> ++++++++++++++++++++++++++++\n'
        if self.rules_ is not None:
            display += f'> SRules --  Number of Rules: {len(self.rules_)}\n'
        if self.minimal_rules_ is not None:
            display += f'> SRules --  Number of Minimal Rules: {len(self.minimal_rules_)}\n'
        display += '> ++++++++++++++++++++++++++++\n'
        return display

    def __str__(self):
        display = self.rules_description()
        display += '> ------------------------------\n'
        # TODO: TO METHODS?
        if self.minimal_rules_ is not None:
            for num in range(len(self.minimal_rules_)):
                display += f'{self.minimal_rules_[num]}'
        else:
            for num in range(len(self.rules_)):
                display += f'{self.rules_[num]}'

        return display
