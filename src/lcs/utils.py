import csv

import numpy as np


def initialize_population():
    """Initialize the empty population. This is only called once at the
    beginning of the cycle."""
    return []


def update_population(classifier, population):
    population.append(classifier)
    return population


def get_data_length(data):
    """Get the length of the file so that the get_instance function doesn't
    return anything if requested line is not present"""
    with open(data, "r") as file:
        return sum(1 for row in file)


def convert_int(instance):
    """Convert the instance into a list of integers"""
    int_instance = []
    for i in instance:
        int_instance.append(int(i))
    return int_instance


def numpy_get_instance(data, line_num):
    _data = np.loadtxt(data, delimiter=",", skiprows=1, dtype=int)
    for row in _data:
        yield row


def get_instance(data, line_num):
    """Create a function that gets the data from a file an returns a specified
    instance of the dataset to the LCS This returns a single training instance
    from the data and does not load the entire data file into memory"""

    lines = get_data_length(data)
    with open(data, "r") as source:
        reader = csv.reader(source)
        if line_num > lines:
            return
        for _ in range(line_num):
            next(reader)
        return convert_int(next(reader))


def does_match(state, instance):
    """Create a does_match function that compares each attribute between two classifiers
    The states of each classifier are tuples of (index, value). Only some indices
    are specified, if they are not, they are equivalent to the hash "don't care"
    symbol"""
    for i in range(len(state)):
        index = state[i][0]
        if state[i][1] != instance[index]:
            return False
    return True
