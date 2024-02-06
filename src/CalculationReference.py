import math
import statistics


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
    return statistics.median(list_of_numbers)


def mode(list_of_numbers: list[float]):
    """
    Returns the mode (most frequent) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Mode value
    """
    return statistics.mode(list_of_numbers)


def variance(list_of_numbers: list[float], mean_value: float):
    """
    Returns the variance from a supplied list of numbers and mean.
    Found by taking the average of squared deviations from the mean

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :type mean_value: float
    :param mean_value: A float assumed to be the mean
    :rtype: float
    :return: The variance
    """
    return sum([(x - mean_value) ** 2 for x in list_of_numbers])


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
    :param z: The score
    :return: Probability distribution(s) for each element in z
    """
    prob_dists = softmax_temperature(z, tau=1.0)
    return prob_dists


def softmax_temperature(z: list[float], tau: float) -> list[float]:
    """
    Computes softmax values for each element in z with temperature represented as tau
    :param z: The sore
    :param tau: The temperature
    :return: Probability distribution(s) for each element in z affected by tau
    """
    z_norm = z
    if tau != 1.0:
        z_norm = list(map(lambda z_i: z_i / tau, z))
    z_exp = list(map(lambda z_i: math.exp(z_i), z_norm))
    prob_dists = list(map(lambda z_e: z_e / sum(z_exp), z_exp))
    return prob_dists


def sigmoid(z: list[float]) -> list[float]:
    """
    Computes sigmoid values for each element in z
    :param z: The scores
    :return: Probability distribution(s) for each element in z
    """
    return list(map(lambda z_i: 1 / (1 + math.exp(-z_i)), z))
