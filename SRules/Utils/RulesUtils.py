import time


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
