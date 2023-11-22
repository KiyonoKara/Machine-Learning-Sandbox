import statistics


def mean(list_of_numbers):
    """
    Returns mean value (average) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Mean value
    """
    return sum(list_of_numbers) / len(list_of_numbers)


def median(list_of_numbers):
    """
    Returns median value (middle) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Median value
    """
    return statistics.median(list_of_numbers)


def mode(list_of_numbers):
    """
    Returns the mode (most frequent) from a list of numbers, optimized using the native function

    :type list_of_numbers: list[float]
    :param list_of_numbers: List of numbers
    :rtype: float
    :return: Mode value
    """
    return statistics.mode(list_of_numbers)


def variance(list_of_numbers, mean_value):
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