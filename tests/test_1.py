from lcs import utils
from lcs.core import covering
from lcs.set_creation import create_correct_set, create_match_set


def testing1(data, specificity):
    population = utils.initialize_population()
    length = utils.get_data_length(data)
    for i in range(1, length):
        instance = utils.get_instance(data, i)
        match_set = create_match_set(population, instance)
        correct_set = create_correct_set(match_set, instance)
        if len(correct_set) == 0:
            classifier = covering(instance, i, specificity=specificity)
            utils.update_population(classifier, population)

    return population
