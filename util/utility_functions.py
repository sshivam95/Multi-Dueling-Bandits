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


def get_run_times_saps() -> np.ndarray:
    """Extract the run times of the SAPS solver.

    Returns
    -------
    np.ndarray
        The run times of all the parameterizations on all the problem instances in the dataset. The shape of this matrix is (time_horizon, num_arms)
    """
    # Read the run times of the SAPS solver from the csv file
    running_times_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "sat"), "Data_saps_swgcp_reduced"),
            "cpu_times_inst_param.csv",
        ),
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


def get_parameterization_saps() -> np.ndarray:
    """Extract the features of the parameters of the SAPS solver for the SAT problem.

    Returns
    -------
    np.ndarray
        The features of the parameterizations of the SAPS solver. The shape of this matrix is (num_arms, 4)
    """
    # Read the parameterization of the SAPS solver from the .txt file
    parametrizations_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "sat"), "Data_saps_swgcp_reduced"),
            "Random_Parameters_SAPS.txt",
        ),
    )
    with open(parametrizations_file, "r") as f:
        lineList = f.readlines()

    # Extract the parameterizations and preprocessing them
    parametrizations = [float(s) for s in re.findall(r"-?\d+\.?\d*", lineList[0])]
    parametrizations = np.reshape(parametrizations, (20, 4))
    parametrizations = preprocessing.normalize(parametrizations)
    return parametrizations


def get_features_saps() -> np.ndarray:
    """Extract the features of the problem instances of the SAT problem.

    Returns
    -------
    np.ndarray
        The preprocessed features of the problem instances of the SAT problem. The shape of this matrix is (time_horizon, 8)
    """
    # read features
    features_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "sat"), "Data_saps_swgcp_reduced"),
            "Reduced_Features_SWGCP_only_5000.csv",
        ),
    )
    features = []
    with open(features_file, newline="") as csvfile:
        features_data = list(csv.reader(csvfile))
    for i in range(1, len(features_data)):
        next_line = features_data[i]
        del next_line[0]
        next_feature_vector = [float(s) for s in next_line]
        features.append(next_feature_vector)
    features = np.asarray(features)

    # Preprocess the features by removing the features with low variance and high correlation
    features = preprocess(
        data=features, variance_threshold=0.001, correlation_threshold=0.98
    )
    return features


def get_run_times_mips() -> np.ndarray:
    """Extract the run times of the CPLEX solver.

    Returns
    -------
    np.ndarray
        The run times of all the parameterizations on all the problem instances in the dataset. The shape of this matrix is (time_horizon, num_arms)
    """
    running_times_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "mips"), "bids_arbitrary_data"),
            "20_Params_times.csv",
        ),
    )
    running_times = pd.read_csv(running_times_file, delimiter="\t")
    running_times = running_times.iloc[:, 1:].to_numpy(dtype=float)
    # lambda_ = 10
    # running_times = np.exp(-lambda_ * running_times)
    return running_times


def get_parameterization_mips() -> np.ndarray:
    """Extract the features of the parameters of the CPLEX solver for the MIPS problem.

    Returns
    -------
    np.ndarray
        The features of the parameterizations of the CPLEX solver. The shape of this matrix is (num_arms, 24)
    """
    parametrizations_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "mips"), "bids_arbitrary_data"),
            "20_params_bids_arb.csv",
        ),
    )
    params_mips = pd.read_csv(parametrizations_file, delimiter="\t")
    params = params_mips.to_numpy()
    parametrizations = preprocess(
        data=params[:, 1:], variance_threshold=0.15, correlation_threshold=0.95
    )
    return parametrizations


def get_features_mips() -> np.ndarray:
    """Extract the features of the problem instances of the MIPS problem.

    Returns
    -------
    np.ndarray
        The preprocessed features of the problem instances of the MIPS problem. The shape of this matrix is (time_horizon, 13)
    """
    # read features
    features_file = os.path.join(
        f"{Path.cwd()}",
        os.path.join(
            os.path.join(os.path.join("Data", "mips"), "bids_arbitrary_data"),
            "Features_1500_inst_sort_bids_arbitrary_MIP_CA.csv",
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
    parametrizations: np.ndarray,
    features: np.ndarray,
    joint_feature_map_mode: str,
    context_feature_dimensions: int,
) -> np.ndarray:
    """Get the context matrix by using joint feature map mode on the parameterizations and features of the problem instances.

    Each row in the matrix corresponds to the problem instance to be solved.
    Each context vector is of dimension `context_feature_dimensions` based on the `joint_feature_map_mode`.

    Parameters
    ----------
    parametrizations : np.ndarray
        The parameters of the solver.
    features : np.ndarray
        The features of the problem instances.
    joint_feature_map_mode : str
        A feature mapping mode.
    context_feature_dimensions : int
        Dimensions of the context features to be formed.

    Returns
    -------
    np.ndarray
        An np.ndarray of shape (number of instances, number of arms, context dimension).
    """
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
    x: np.ndarray, y: np.ndarray, mode: Optional[str] = "polynomial"
) -> np.ndarray:
    """
    The feature engineering part of the CPPL algorithm.

    Parameters
    ----------
    x : np.ndarray
        Features of problem instances.
    y : np.ndarray
        Features of parameterization.
    mode : str, optional
        Mode of the solver, by default "polynomial"

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
        num = num + (context_vector[arm] * np.exp(np.dot(theta, context_vector[arm])))
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
        t_1 = t_1 + (context_matrix[arm] * np.exp(np.dot(theta, context_matrix[arm])))
    num_1 = np.outer(t_1, t_1)
    denominator_1 = 0
    for arm in selection:
        denominator_1 = denominator_1 + np.exp(np.dot(theta, context_matrix[arm])) ** 2
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


def stochastic_gradient_descent(
    theta: np.ndarray,
    gamma_t: float,
    selection: np.ndarray,
    context_vector: np.ndarray,
    winner: int,
) -> np.ndarray:
    """Calculate and update the weight parameters using stochastic gradient descent.

    Parameters
    ----------
    theta : np.ndarray
        The weight parameters from the previous time step.
    gamma_t : float
        The learning rate.
    selection : np.ndarray
        The subset of arms selected from the pool of arms.
    context_vector : np.ndarray
        The context vector in the current time step.
    winner : int
        The winner arm in the selected subset. The winner arm is the one which has the lowest run time to solve the given problem instance.

    Returns
    -------
    np.ndarray
        The updated estimated weight parameters.
    """
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


def preprocess(
    data: np.ndarray, variance_threshold: float, correlation_threshold: float
) -> np.ndarray:
    """A utility function to preprocess the data.

    Parameters
    ----------
    data : np.ndarray
        The input raw data.
    variance_threshold : float
        The variance threshold to drop the data points lower than this threshold.
    correlation_threshold : float
        The correlation threshold to drop the data points higher than this threshold.

    Returns
    -------
    np.ndarray
        The processes and clean data.
    """
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
