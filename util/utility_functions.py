"""Implementation of various helper functions."""
import csv
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn import preprocessing


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


def random_sample_from_list(array, random_state, size: int, exclude: Optional[int] = None):
    if exclude is not None:
        return random_state.choice(
            list(np.delete(array, exclude)), size=size, replace=False
        )
    else:
        return random_state.choice(
            list(array), size=size, replace=False
        )


def exclude_elements(array, exclude):
    return np.array(list(set(array) - set(exclude)))


def read_run_times():
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
    lambda_ = 100
    running_times = np.exp(-lambda_ * running_times)

    return running_times


def read_parameterization():
    parametrizations_file = "Data_saps_swgsp\\Random_Parameters_SAPS.txt"
    with open(parametrizations_file) as f:
        lineList = f.readlines()
    f.close()
    parametrizations = [float(s) for s in re.findall(r"-?\d+\.?\d*", lineList[0])]
    parametrizations = np.reshape(parametrizations, (20, 4))
    parametrizations = preprocessing.normalize(parametrizations)

    return parametrizations
