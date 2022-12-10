"""Implementation of various helper functions."""
import csv
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures


def argmin_set(
    array: np.array, exclude_indexes: Optional[List[int]] = None
) -> List[int]:
    """Calculate the complete argmin set, returning an array with all indices.

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with minimum value from remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmin should be calculated.
    exclude_indexes
        Indices to exclude in the argmin operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the minimum value.
    """
    # np.argmin only returns the first index, to get the whole set,
    # we first find the minimum and then search for all indices which point
    # to a value equal to this minimum
    min_value = 0
    result: List[int] = []
    for idx, item in enumerate(array):
        if exclude_indexes is not None and idx in exclude_indexes:
            continue
        if not np.ma.getmaskarray(array)[idx] and (
            len(result) == 0 or item < min_value
        ):
            min_value = item
            result = [idx]
        elif item == min_value:
            result.append(idx)
    return np.array(result)


def argmax_set(
    array: np.array, exclude_indexes: Optional[List[int]] = None
) -> List[int]:
    """Calculate the complete argmax set, returning an array with all indices..

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with maximum value from the remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmax should be calculated
    exclude_indexes
        Indices to exclude in the argmax operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the maximum value.
    """
    # np.argmax only returns the first index, to get the whole set,
    # we first find the maximum and then search for all indices which point
    # to a value equal to this maximum
    max_value = 0
    result: List[int] = []
    for idx, item in enumerate(array):
        if exclude_indexes is not None and idx in exclude_indexes:
            continue
        if not np.ma.getmaskarray(array)[idx] and (
            len(result) == 0 or item > max_value
        ):
            max_value = item
            result = [idx]
        elif item == max_value:
            result.append(idx)
    return np.array(result)


def pop_random(
    input_list: List[int], random_state: np.random.RandomState, amount: int = 1
) -> List[int]:
    """Remove randomly chosen elements from a given list and return them.

    If the list contains less than or exactly ``amount`` elements, all elements are chosen.

    Parameters
    ----------
    input_list
        The list from which an arm should be removed.
    random_state
        The random state to use.
    amount
        The number of elements to pick, defaults to ``1``.

    Returns
    -------
    List[int]
        The list containing the removed elements.
    """
    if len(input_list) <= amount:
        list_copy = input_list.copy()
        input_list.clear()
        return list_copy
    else:
        picked = []
        left_over = input_list
        for _ in range(amount):
            random_index = random_state.randint(0, len(input_list))
            removed_element = left_over.pop(random_index)
            picked.append(removed_element)
        return picked


def random_sample_from_list(
    array, random_state, size: int, exclude: Optional[int] = None
):
    if exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
        return random_state.choice(
            exclude_elements(array=array, exclude=exclude),
            size=size,
            replace=False,
        )
    else:
        return random_state.choice(list(array), size=size, replace=False)


def exclude_elements(array, exclude):
    return np.array(list(set(array) - set(exclude)))


def get_run_times_saps():
    running_times_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join("Data_saps_swgcp_reduced", "cpu_times_inst_param.csv"),
    )
    running_times = []
    with open(running_times_file, newline="") as csvfile:
        running_times_data = list(csv.reader(csvfile))
    for i in range(1, len(running_times_data)):
        next_line = running_times_data[i][0]
        next_rt_vector = [float(s) for s in re.findall(r"-?\d+\.?\d*", next_line)][2:]
        running_times.append(next_rt_vector)
    running_times = np.asarray(running_times)
    # lambda_ = 10
    # running_times = np.exp(-lambda_ * running_times)
    return running_times


def get_parameterization_saps():
    parametrizations_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join("Data_saps_swgcp_reduced", "Random_Parameters_SAPS.txt"),
    )
    with open(parametrizations_file, "r") as f:
        lineList = f.readlines()
    parametrizations = [float(s) for s in re.findall(r"-?\d+\.?\d*", lineList[0])]
    parametrizations = np.reshape(parametrizations, (20, 4))
    parametrizations = preprocessing.normalize(parametrizations)

    return parametrizations


def get_features_saps():
    # read features
    features_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join("Data_saps_swgcp_reduced", "Reduced_Features_SWGCP_only_5000.csv"),
    )
    features = []
    # problem_instances = []
    with open(features_file, newline="") as csvfile:
        features_data = list(csv.reader(csvfile))
    for i in range(1, len(features_data)):
        next_line = features_data[i]
        # problem_instances.append(next_line[0])
        del next_line[0]
        next_feature_vector = [float(s) for s in next_line]
        features.append(next_feature_vector)
    features = np.asarray(features)
    features = preprocess(
        data=features, variance_threshold=0.001, correlation_threshold=0.98
    )
    return features


def get_run_times_mips():
    running_times_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join("bids_arbitrary_data", "20_Params_times.csv"),
    )
    running_times = pd.read_csv(running_times_file, delimiter="\t")
    running_times = running_times.iloc[:, 1:].to_numpy(dtype=float)
    # lambda_ = 10
    # running_times = np.exp(-lambda_ * running_times)
    return running_times


def get_parameterization_mips():
    parametrizations_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join("bids_arbitrary_data", "20_params_bids_arb.csv"),
    )
    params_mips = pd.read_csv(parametrizations_file, delimiter="\t")
    params = params_mips.to_numpy() 
    parametrizations = preprocess(
        data=params[:, 1:], variance_threshold=0.15, correlation_threshold=0.95
    )
    return parametrizations


def get_features_mips():
    # read features
    features_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            "bids_arbitrary_data", "Features_1500_inst_sort_bids_arbitrary_MIP_CA.csv"
        ),
    )
    features = []
    with open(features_file, newline="") as csvfile:
        features_data = list(csv.reader(csvfile))
    for i in range(1, len(features_data)):
        next_line = features_data[i]
        next_feature_vector = [
            float(s) for s in re.findall(r"-?\d+\.?\d*", next_line[0])
        ]
        features.append(next_feature_vector)
    features = np.asarray(features)
    features = features[:, 2:]
    features = preprocess(
        data=features, variance_threshold=0.01, correlation_threshold=0.9
    )
    return features


def get_context_matrix(
    parametrizations, features, joint_feature_map_mode, context_feature_dimensions
):
    n_arms = parametrizations.shape[0]
    min_max_scaler = preprocessing.MinMaxScaler()
    context_matrix = []
    for t in range(features.shape[0]):
        X = np.zeros((n_arms, context_feature_dimensions))
        next_context = features[t, :]
        for i in range(n_arms):
            # next_param = parametrizations[i]
            X[i, :] = join_feature_map(
                x=parametrizations[i], y=next_context, mode=joint_feature_map_mode
            )
        X = preprocessing.normalize(X)
        X = min_max_scaler.fit_transform(X)
        context_matrix.append(X)
    context_matrix = np.array(context_matrix)
    return context_matrix


def join_feature_map(
    x: np.ndarray, y: np.ndarray, mode: Optional[str] = "polynimial"
) -> np.ndarray:
    """
    The feature engineering part of the CPPL algorithm.

    Parameters
    ----------
    x : np.ndarray
        Features of problem instances.
    y : np.ndarray
        Features of parameterization.
    mode : str
        Mode of the solver.

    Returns
    -------
    np.ndarray
        A numpy array of the transforms joint features based on the mode of the solver.
    """
    if mode == "concatenation":
        return np.concatenate((x, y), axis=0)
    elif mode == "kronecker":
        return np.kron(x, y)
    elif mode == "polynomial":
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        return poly.fit_transform(np.concatenate((x, y), axis=0).reshape(1, -1))


def get_round_winner(
    running_time, arm_1: Optional[int] = None, arm_2: Optional[int] = None
):
    if arm_1 and arm_2 is None:
        return np.argmax(running_time)
    elif arm_1 != arm_2:
        winner = arm_1 if check_runtime(arm_1, arm_2, running_time) else arm_2
        return winner


def check_runtime(arm_i, arm_j, running_time):
    if running_time[arm_i] < running_time[arm_j]:
        return False
    else:
        return True


def gradient(
    theta: np.ndarray,
    winner_arm: int,
    selection: np.ndarray,
    context_vector: np.ndarray,
) -> float:
    """
    Calculate the gradient of the log-likelihood function in the partial winner feedback scenario.

    Parameters
    ----------
    theta : np.ndarray
        Score or weight parameter of each arm in the contender pool. Theta is use to calculate the log-linear estimated skill parametet v_hat.
    winner_arm : int
        Winner arm (parameter) in the subset.
    subset : np.ndarray
        A subset of arms from the contender pool for solving the instances.
    context_matrix : np.ndarray
        A context matrix where each element is associated with one of the different arms and contains the
        properties of the arm itself as well as the context in which the arm needs to be chosen.

    Returns
    -------
    res: float
        The gradient of the log-likelihood function in the partial winner feedback scenario.
    """
    denominator = 0
    num = np.zeros((len(theta)))
    for arm in selection:
        denominator = denominator + np.exp(np.dot(theta, context_vector[arm]))
        num = num + (
            context_vector[arm] * np.exp(np.dot(theta, context_vector[arm]))
        )
    res = context_vector[winner_arm] - (num / denominator)
    return res.reshape(
        -1,
    )


def hessian(
    theta: np.ndarray, selection: np.ndarray, context_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate the hessian matrix of the log-likelihood function in the partial winner feedback scenario.

    Parameters
    ----------
    theta : np.ndarray
        Score parameter matrix where each row represents each arm in the contender pool. Theta is use to calculate the log-linear estimated skill parametet v_hat.
    subset : np.ndarray
        A subset of arms from the contender pool for solving the instances.
    context_matrix : np.ndarray
        A context matrix where each element is associated with one of the different arms and contains
        the properties of the arm itself as well as the context in which the arm needs to be chosen.

    Returns
    -------
    np.ndarray
        A hessian matrix of the log-likelihood function in the partial winner feedback scenario.
    """
    dimension = len(theta)
    t_1 = np.zeros(dimension)
    for arm in selection:
        t_1 = t_1 + (
            context_matrix[arm] * np.exp(np.dot(theta, context_matrix[arm]))
        )
    num_1 = np.outer(t_1, t_1)
    denominator_1 = 0
    for arm in selection:
        denominator_1 = (
            denominator_1 + np.exp(np.dot(theta, context_matrix[arm])) ** 2
        )
    s_1 = num_1 / denominator_1
    num_2 = 0
    for j in selection:
        num_2 = num_2 + (
            np.exp(np.dot(theta, context_matrix[j]))
            * np.outer(context_matrix[j], context_matrix[j])
        )
    denominator_2 = 0
    for arm in selection:
        denominator_2 = denominator_2 + np.exp(np.dot(theta, context_matrix[arm]))
    s_2 = num_2 / denominator_2
    return s_1 - s_2


def stochastic_gradient_descent(theta, gamma_t, selection, context_vector, winner):
    derivative = gradient(
        theta=theta,
        winner_arm=winner,
        selection=selection,
        context_vector=context_vector,
    )
    theta += gamma_t * derivative
    theta[theta < 0] = 0
    theta[theta > 1] = 1
    return theta


def preprocess(data, variance_threshold, correlation_threshold):
    # normalize#########
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    # Drop Highly Correlated Features #######
    df = pd.DataFrame(data)
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [
        column for column in upper.columns if any(upper[column] > correlation_threshold)
    ]
    # Drop features
    df.drop(df[to_drop], axis=1, inplace=True)
    data = df.to_numpy()
    # Drop features with lower variance
    selector = VarianceThreshold(variance_threshold)
    data = selector.fit_transform(data)
    ########
    return data
