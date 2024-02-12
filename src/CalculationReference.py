import math
from typing import Union
from collections import Counter

# Population encompasses the all the data
# Sample is a subset of population
group_dict = {
    'population': 0,
    'sample': 1
}


def mean(list_of_numbers: list[float]):
    """
    Returns mean value (average) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Mean value
    """
    return sum(list_of_numbers) / len(list_of_numbers)


def median(list_of_numbers: list[float]):
    """
    Returns median value (middle) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Median value
    """
    length = len(list_of_numbers)
    if length == 0:
        raise ValueError("List must have at least one element")
    elif length == 1:
        return list_of_numbers[0]
    sorted_nums = sorted(list_of_numbers)
    middle = length // 2
    if not length % 2:
        return (sorted_nums[middle - 1] + sorted_nums[middle]) / 2.0
    return sorted_nums[middle]


def mode(list_of_numbers: list[float]) -> float:
    """
    Returns the mode (most frequent numbers) from a list of numbers

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Mode value
    """
    return max(set(list_of_numbers), key=list_of_numbers.count)


def mode_v2(list_of_numbers: list[float]) -> Union[float, list[float]]:
    """
    Returns the mode(s) (most frequent numbers) from a list of numbers
    May return a list if there are multiple modes, otherwise it's a single float
    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: Union[float, list[float]]
    :return: Mode value(s)
    """
    counts = Counter(list_of_numbers)
    max_frequency = max(counts.values())
    modes = [n for n, frequency in counts.items() if frequency == max_frequency]
    return modes[0] if len(modes) == 1 else modes


def variance(list_of_numbers: list[float], mean_value: float, grouping_type='population'):
    """
    Returns the variance from a supplied list of numbers and mean.
    Found by taking the average of squared deviations from the mean

    :param grouping_type: Type of data, sample or population
    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :type mean_value: float
    :param mean_value: A float assumed to be the mean
    :rtype: float
    :return: The variance
    """
    # Get number to subtract from N
    # Population: N - 0 = N
    # Sample: N - 1
    n_0 = group_dict.get(grouping_type, 0)

    return sum([(x - mean_value) ** 2 for x in list_of_numbers]) / (len(list_of_numbers) - n_0)


def covariance(list_x: list[float], mean_x: float, list_y: list[float], mean_y: float):
    """
    Returns the covariance between two lists of numbers and uses both of their means

    :type list_x: list[float]
    :param list_x: List of numbers
    :type mean_x: float
    :param mean_x: The mean of 'x' list of numbers
    :type list_y: list[float]
    :param list_y: List of numbers
    :type mean_y: float
    :param mean_y: The mean of 'y' list of numbers
    :rtype: float
    :return: The covariance
    """
    return sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(list_x, list_y))


def softmax(z: list[float]) -> list[float]:
    """
    Computes softmax values for each element in z
    Makes call on softmax_temperature by setting tau to 1.0 and producing the origin
    softmax distribution
    :param z: The scores
    :return: Probability distributions for each element in z
    """
    prob_dists = softmax_temperature(z, tau=1.0)
    return prob_dists


def softmax_temperature(z: list[float], tau: float) -> list[float]:
    """
    Computes softmax values for each element in z with temperature represented as tau
    :param z: The scores
    :param tau: The temperature
    :return: Probability distributions for each element in z affected by tau
    """
    z_norm = z
    if tau != 1.0:
        z_norm = list(map(lambda z_i: z_i / tau, z))
    z_exp = list(map(lambda z_i: math.exp(z_i), z_norm))
    prob_dists = list(map(lambda z_e: z_e / sum(z_exp), z_exp))
    return prob_dists


def sigmoid(z: Union[float, list[float]]) -> Union[float, list[float]]:
    """
    Computes sigmoid values for each element in z
    :param z: The scores
    :return: Probability distribution(s) for each element in z
    """
    if isinstance(z, float) or isinstance(z, int):
        return 1 / (1 + math.exp(-z))
    return list(map(lambda z_i: 1 / (1 + math.exp(-z_i)), z))


def relu(x: float) -> float:
    """
    Computes ReLU value for the supplied number
    :param x: The float
    :return: The ReLU result
    """
    return x * (x > 0.0)


def stdev(nums: list[float], grouping_type='population') -> float:
    """
    Computes standard deviation given a list of floats
    Returns 0 if only one number is provided
    Defaults to population standard deviation
    :param nums: List of floats
    :param grouping_type: Grouping type, population or sample
    :return: The standard deviation
    """
    if len(nums) <= 1:
        return 0
    # Get number to subtract from N
    # Population: N - 0 = N
    # Sample: N - 1
    p = group_dict.get(grouping_type, 0)
    # Mean
    m = mean(nums)
    # Variance
    v = variance(nums, m) / (len(nums) - p)
    return math.sqrt(v)


def dot_product(v1: list[float], v2: list[float]) -> float:
    """
    Computes dot product of two vectors as lists
    :param v1: First vector
    :param v2: Second vector
    :return:
    """
    if len(v1) != len(v2):
        raise ValueError(f'Vectors must have same dimensions, got ({len(v1)}) and ({len(v2)})')
    return sum([v_i * v_j for v_i, v_j in zip(v1, v2)])
