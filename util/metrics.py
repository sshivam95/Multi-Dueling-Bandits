import numpy as np

from util.utility_functions import argmin_set


def regret_preselection(skill_vector, selection):
    selection_set_size = len(selection)
    # average performance of best set
    # best_item = np.argmax(skill_vector)
    best_items = argmin_set(skill_vector)
    mask = np.isin(best_items, selection)
    if True in mask:
        return 0
    # if best_item in selection:
    #     return 0
    else:
        S_star = (-skill_vector).argsort()[0:selection_set_size]
        S_star_perf = 0
        S_perf = 0
        for i in range(selection_set_size):
            S_star_perf = S_star_perf + average_performance(
                arm_in_selection=S_star[i], skill_vector=skill_vector
            )
            S_perf = S_perf + average_performance(
                arm_in_selection=selection[i], skill_vector=skill_vector
            )
        return (S_star_perf - S_perf) / selection_set_size


def average_performance(arm_in_selection, skill_vector):
    result = 0
    v_i = skill_vector[arm_in_selection]
    n = len(skill_vector)
    for j in range(n):
        if j != arm_in_selection:
            v_j = skill_vector[j]
            if v_i == 0 and v_j == 0:
                result = result
            else:
                result = result + (v_i / (v_i + v_j))
    result = result / (n - 1)
    return result


def compute_cumulative_regret(regrets):
    cummulative_regrets = []
    for regret in regrets:
        cummulative_regret = [0]
        for i, value in enumerate(regret):
            cummulative_regret.append((cummulative_regret[-1] + value))
        cummulative_regrets.append(cummulative_regret)
    return np.array(cummulative_regrets)
