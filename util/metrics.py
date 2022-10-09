def average_performance_saps(i, v):
    res = 0
    v_i = v[i]
    n = len(v)
    for j in range(n):
        if j != i:
            v_j = v[j]
            res = res + (v_i / (v_i + v_j))
    res = res / (n - 1)
    return res


def compute_cumm_reg(regrets):
    cummulative_regrets = []
    for regret in regrets:
        cummulative_regret = [0]
        for i, value in enumerate(regret):
            cummulative_regret.append((cummulative_regret[-1] + value))
        cummulative_regrets.append(cummulative_regret)
    return cummulative_regrets[1:]