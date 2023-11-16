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
    for idx, current_rule in reversed(list(enumerate(sorted_rules))):
        current_full_rule = current_rule.get_full_rule()
        should_include = True
        for new_index in range(len(sorted_rules) - 1, idx, -1):
            new_current_full_rule = sorted_rules[new_index].get_full_rule()
            if current_full_rule in new_current_full_rule:
                # no valida
                should_include = False
                break
        if should_include:
            minimal_rules.append(current_rule)

    elapsed_time = time.time() - start_time
    if display_logs:
        print(f"Elapsed time to compute the prune_rules: {elapsed_time:.3f} seconds")

    return minimal_rules.reverse()
