import time
import copy
from sklearn.base import ClassifierMixin
import SRules.Utils.QueryUtils as q
from SRules.FeatureComparer import FeatureComparer
from SRules.Node import Node
import SRules.Utils.FeatureImportance as FeatureImportance
from SRules.Utils import RulesUtils
from SRules.Tests.Utils import DatasetUtils


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
                 p_significance=0.95,
                 minImp=0.2,
                 min_accuracy_coefficient=0.9,
                 minInsNode=3
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
        self.p_significance = p_significance
        self.minImp = minImp
        self.min_accuracy_coefficient = min_accuracy_coefficient
        self.minInsNode = minInsNode
        self.display_features = display_features
        self.display_logs = display_logs
        self.recursive = recursive
        self.all_rules_ = []

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
            if node_values_total >= self.minInsNode:
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
                self.minImp,
                self.display_logs,
                self.display_features)

        if self.most_important_features_ is None or []:
            return None, None, None

        # Generate Tree

        _, minimal_dataset = self.generate_tree(dataset=dataset)
        return self.nodes_dict, minimal_dataset, self.most_important_features_

    def generate_tree(self, dataset):
        # Genera el árbol binario y obtiene las combinaciones que indican que hay un patrón:

        if self.most_important_features_ is [] or None or len(self.most_important_features_) == 0:
            return False, dataset

        minimal_dataset = self.minimal_dataset(dataset)

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
                self.minImp,
                self.display_logs,
                self.display_features)

        if self.most_important_features_ is None or []:
            return

        # Lista de nodos válidos
        self.pattern_list_valid_nodes.extend(RulesUtils.obtain_pattern_list_of_valid_nodes_with_p_value(self.nodes_dict,
                                                                                                        self.p_significance,
                                                                                                        self.feature_names,
                                                                                                        self.display_logs))

        if self.most_important_features_ is not None or []:
            minimal_dataset = self.minimal_dataset(dataset)

        if minimal_dataset is None or []:
            return

        # Categoriza patrones
        self.rules_.extend(RulesUtils.categorize_patterns(minimal_dataset,
                                                          self.pattern_list_valid_nodes,
                                                          self.min_accuracy_coefficient,
                                                          self.target_class_positive,
                                                          self.target_class_negative,
                                                          self.target_true,
                                                          self.target_false,
                                                          self.display_logs))

        # Sort rules AND Prune rules
        self.minimal_rules_ = RulesUtils.prune_rules(self.minimal_rules_,
                                                     self.sorting(sorting_method),
                                                     self.display_logs)

        self.all_rules_.append(self.minimal_rules_)

        return self

    def minimal_dataset(self, dataset):
        minimal_dataset = copy.deepcopy(dataset[self.most_important_features_ + [self.target_value_name]])
        minimal_dataset.sort_values(self.most_important_features_, inplace=True, ascending=True)
        return minimal_dataset

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
        if self.display_logs:
            print("INIT")

        dataset = copy.deepcopy(original_dataset)
        X_train = copy.deepcopy(X_train)
        y_train = copy.deepcopy(y_train)

        self.feature_importance = FeatureImportance.extract_feature_importance(method, X_train, use_shap, use_lime)

        if self.display_logs:
            print(" -> TRAINING MODEL")

        if self.recursive is False:
            if self.display_logs:
                print(" --> SINGLE FIT")
            return self.single_fit(dataset=dataset,
                                   feature_importance=self.feature_importance,
                                   sorting_method="target_accuracy")

        if self.display_logs:
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

        if self.display_logs:
            print(" ---> Fitting Recursive Model")
        while True:
            if self.display_logs:
                print(f" ----> Step: {recursive_counter}")
            self.single_fit(dataset,
                            feature_importance=feature_importance,
                            sorting_method="target_accuracy")

            if self.display_logs:
                print(f" ----> {recursive_counter} - Model Prediction")
            # predict
            y_pred_train_rules = self.predict(X_train, sorting_method)

            # New datasets
            if self.display_logs:
                print(f" ----> {recursive_counter} - Creating new dataset")
            X_train, y_train, dataset, new_len = (DatasetUtils
                                                  .new_datasets(X_train=X_train,
                                                                y_train=y_train,
                                                                y_pred_train_rules=y_pred_train_rules,
                                                                dataset=dataset))

            if self.display_logs:
                print(f' ----> {recursive_counter} - Previous dataset length: {previous_dataset_len}')
                print(f' ----> {recursive_counter} - New dataset length: {new_len}')

            if self.display_logs:
                print(self.rules_description())

            # No improvement or
            if new_len == 0 or previous_dataset_len == new_len:
                if self.display_logs:
                    print(" ---> Finishing fitting model")
                break
            previous_dataset_len = new_len

            if self.display_logs:
                print(f" ----> {recursive_counter} - Fitting method")
            method.fit(X_train, y_train)

            if self.display_logs:
                print(f" ----> {recursive_counter} - Feature importance")
            feature_importance = FeatureImportance.extract_feature_importance(method, X_train, use_shap, use_lime)

            # Clean variables
            self.clean()
            recursive_counter += 1

        # Join Rules
        if self.display_logs:
            print(" ---> Joining all rules")
        self.rules_ = RulesUtils.join_all_rules(self.all_rules_)
        # Prune rules
        if self.display_logs:
            print(" ---> Pruning rules")
        self.minimal_rules_ = RulesUtils.prune_rules(None,
                                                     self.sorting(),
                                                     self.display_logs)

        if self.display_logs:
            print("END")
        return self

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
            if self.display_logs:
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

    def __str__(self):
        display = RulesUtils.rules_description(self.rules_, self.minimal_rules_)
        display += RulesUtils.print_minimal_rules(self.minimal_rules_)
        display += RulesUtils.print_rules(self.rules_)
        return display
