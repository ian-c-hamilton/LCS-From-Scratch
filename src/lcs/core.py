import copy
import csv
import math
import pprint
import random
from pathlib import Path


def initialize_population():
    """Initialize the empty population. This is only called once at the
    beginning of the cycle."""
    return []


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


def create_match_set(population, instance):
    """Create the match set by comparing the attributes of each classifier in
    the population with the current instance"""
    match_set = []
    if len(population) == 0:
        # Returns an empty patch set if the population length is zero, this is
        # importance for covering
        return match_set
    else:
        # Is a slice of population necessary? This then iterates on the copy of a list,
        # instead of the original. Do I need to return the edited populataion copy?
        for classifier in population:
            state = classifier["state"]
            if does_match(state, instance) == True:
                match_set.append(classifier)
                classifier["match count"] += 1
                classifier["accuracy"] = (
                    classifier["correct count"] / classifier["match count"]
                )
                classifier["fitness"] = classifier["accuracy"] ** 5
        return match_set


def create_correct_set(match_set, instance):
    # Create the correct set by comparing the class or action of each classifier in the match set with the current instance
    correct_set = []
    # Similar to the match set, return an empty correct set if the match set is empty
    if len(match_set) == 0:
        return correct_set
    else:  # Is the slice of match set necessary? Do I need to return a new match set?
        for classifier in match_set:
            # This assumes that the classification is the last item in the training
            # instance list. Most of the time this is the case
            if classifier["action"] == instance[-1]:
                correct_set.append(classifier)
                classifier["correct count"] += 1
                classifier["accuracy"] = (
                    classifier["correct count"] / classifier["match count"]
                )
                classifier["fitness"] = classifier["accuracy"] ** 5
        return correct_set


def covering(instance, iteration, specificity):
    """Create a dictionary item to represent the current instance if the
    correct set is empty."""
    state = []
    action = instance[-1]
    for x in range(len(instance) - 1):
        if random.random() < specificity:
            state.append(tuple((x, instance[x])))
    classifier = {
        "state": state,
        "action": action,
        "numerosity": 1,
        "match count": 1,
        "correct count": 1,
        "accuracy": 1,
        "fitness": 1,
        "deletion vote": 1,
        "birth iteration": iteration,
    }
    return classifier


def update_population(classifier, population):
    population.append(classifier)
    return population


def testing1(data, specificity):
    population = initialize_population()
    length = get_data_length(data)
    for i in range(1, length):
        instance = get_instance(data, i)
        match_set = create_match_set(population, instance)
        correct_set = create_correct_set(match_set, instance)
        if len(correct_set) == 0:
            classifier = covering(instance, i, specificity=specificity)
            update_population(classifier, population)
    return population


def tournament_selection(correct_set, tournament_size):
    """Create a function that takes in a set, like the correct set, and selects
    two parent classifiers"""

    tournament = random.choices(correct_set, k=tournament_size)
    return tournament


def parent_selection(tournament):
    """Create a function that selects a parent from the tournament"""
    max_fitness = 0
    parent_index = 0
    for i in tournament:
        if i["fitness"] > max_fitness:
            max_fitness = i["fitness"]
            parent_index = tournament.index(i)
    return tournament[parent_index]


def crossover(parent1, parent2, birth_iteration):

    parent1_attributes = parent1["state"]
    parent2_attributes = parent2["state"]
    # Assumes that crossover only takes place on the correct set, thus both parents have
    # the same action
    action = parent1["action"]
    offspring1_attributes = []
    offspring2_attributes = []
    if len(parent1_attributes) == 0 and len(parent2_attributes) == 0:
        largest_index = 0
    elif len(parent1_attributes) == 0:
        largest_index = parent2_attributes[-1][0]
    elif len(parent2_attributes) == 0:
        largest_index = parent1_attributes[-1][0]
    elif parent1_attributes[-1][0] >= parent2_attributes[-1][0]:
        largest_index = parent1_attributes[-1][0]
    else:
        largest_index = parent2_attributes[-1][0]
    # Use largest index minus 1 or else there is no crossover if the point is equal to
    # largest index
    # The 4 for loops seem excessive, but it keeps attributes in order by index value
    crossover_point = random.randint(0, (largest_index - 1)) if largest_index > 0 else 0
    for i in parent1_attributes:
        if i[0] <= crossover_point:
            offspring1_attributes.append(i)
    for i in parent2_attributes:
        if i[0] <= crossover_point:
            offspring2_attributes.append(i)
    for i in parent1_attributes:
        if i[0] > crossover_point:
            offspring2_attributes.append(i)
    for i in parent2_attributes:
        if i[0] > crossover_point:
            offspring1_attributes.append(i)
    offspring1 = {
        "state": offspring1_attributes,
        "action": action,
        "numerosity": 1,
        "match count": 1,
        "correct count": 1,
        "accuracy": 1,
        "fitness": 1,
        "deletion vote": 1,
        "birth iteration": birth_iteration,
    }
    offspring2 = {
        "state": offspring2_attributes,
        "action": action,
        "numerosity": 1,
        "match count": 1,
        "correct count": 1,
        "accuracy": 1,
        "fitness": 1,
        "deletion vote": 1,
        "birth iteration": birth_iteration,
    }

    return offspring1, offspring2


def mutation(classifier, instance, rate):
    # Split the mutation rate in half to determine if deleting an attribute or adding
    # one. Thus, mutation only happens once for each attribute
    half_rate = rate / 2
    index = 0
    for j in range(len(instance) - 1):
        rand = random.random()
        # Checks to make sure that instance isn't already in the state, preventing duplicates
        if (
            rand >= half_rate
            and rand <= rate
            and (index, instance[j]) not in classifier["state"]
        ):
            classifier["state"].append((index, instance[j]))
        index += 1
    # classifier['state'] = list(set(classifier['state'])) # Turns the state into a set and removes duplicates.
    for i in classifier["state"][:]:  # Again, does the slice matter here?
        rand = random.random()
        # Rate divided by two to differentiate between deletion and specialization of attributes
        if rand < half_rate:
            # randomly delete the specified attribute from the classifier state
            classifier["state"].remove(i)
    # Sorts the tuples by index, making sure that they are always in order.
    classifier["state"].sort()
    return classifier


def more_general(parent, offspring):
    for i in parent["state"]:
        # Simply checks to see if the parent has fewer specified attributes than the
        # child, as long as the specified ones are in the child
        if i not in offspring["state"]:
            return False
    return True


def subsumption(classifier, population):
    # Do you need to loop through the whole population here? Or can you just update the parent parameters
    for i in population:
        if i["state"] == classifier["state"] and i["action"] == classifier["action"]:
            i["numerosity"] += 1
            # i['deletion vote'] = i['numerosity'] / i['fitness']
            i["deletion vote"] = 1 / i["fitness"]
    return


def already_in(offspring, population):
    """Check to see if the offspring is already in the population"""
    for i in population:
        if i["state"] == offspring["state"] and i["action"] == offspring["action"]:
            return True
    return False


def set_subsumption(population):
    population_copy = copy.deepcopy(population)
    for i in population_copy:
        for j in population_copy:
            if (
                more_general(i, j)
                and i["state"] != j["state"]
                and i["accuracy"] >= j["accuracy"]
            ):
                i["numerosity"] += j["numerosity"]
                if j in population:
                    population.remove(j)
    return


def genetic_algorithm(
    population,
    correct_set,
    tournament_size_fraction,
    mutation_rate,
    training_instance,
    birth_iteration,
):

    # Create tournaments from correct set with size equal to a percentage of the correct set size
    tournament1 = tournament_selection(
        correct_set, math.ceil(tournament_size_fraction * len(correct_set))
    )  # Rounds the tournament size up to a whole number if the correct set is small
    tournament2 = tournament_selection(
        correct_set, math.ceil(tournament_size_fraction * len(correct_set))
    )
    # Select parents from the two tournaments
    parent1 = parent_selection(tournament1)
    parent2 = parent_selection(tournament2)
    # Cross over the parents and produce two offspring
    offspring1, offspring2 = crossover(parent1, parent2, birth_iteration)
    # Mutate the offspring based off the mutation rate
    offspring1 = mutation(offspring1, training_instance, mutation_rate)
    offspring2 = mutation(offspring2, training_instance, mutation_rate)
    # Check if each parent is more general than each child, if so, subsume the child
    if more_general(parent1, offspring1):
        subsumption(parent1, population)
    elif more_general(parent2, offspring1):
        subsumption(parent2, population)
    elif already_in(offspring1, population):
        subsumption(offspring1, population)
    else:  # If the child is not subsumed by either parent, and not already in the population add it to the population
        population.append(offspring1)
    if more_general(parent1, offspring2):
        subsumption(parent1, population)
    elif more_general(parent2, offspring2):
        subsumption(parent2, population)
    elif already_in(offspring2, population):
        subsumption(offspring2, population)
    else:
        population.append(offspring2)
    return


def deletion(population, max_size):
    cumulative_numerosity = 0
    for i in population:  # Loop through the population and sum all the numerosities
        cumulative_numerosity += i["numerosity"]
    if cumulative_numerosity <= max_size:
        return  # if the cumulative numerosity is less than the allowable size, then no deletion occurs

    # Continue deletion until the cumulative numerosity is less than or equal to the max size
    while cumulative_numerosity > max_size:
        # Sort the population based off deletion vote, the highest will be at the front
        population.sort(key=lambda d: d["deletion vote"], reverse=True)
        # If the numerosity of the highest voted classifier is greater than 1, decrease its numerosity by 1
        if population[0]["numerosity"] > 1:
            population[0]["numerosity"] -= 1
            # population[0]['deletion vote'] = population[0]['numerosity'] / population[0]['fitness'] #Update the deletion vote of the first classifier
            population[0]["deletion vote"] = 1 / population[0]["fitness"]
        else:
            # If the numerosity is 1, simply remove the classifier from the population
            population.pop(0)
        cumulative_numerosity = 0  # Reset the cumulative numerosity to 0
        for i in population:
            # Calculate the cumulative numerosity again and loop back up to the while loop
            cumulative_numerosity += i["numerosity"]
    return


def compaction(population, accuracy_cutoff):
    # This should be a simple function, but for some reason, seems to do nothing
    # this slice is important to copy the population and remove items while iterating over it
    for i in population[:]:
        if i["accuracy"] < accuracy_cutoff:
            population.remove(i)
    set_subsumption(population)
    return


def binaryLCS(
    data,
    covering_specificity,
    tournament_size_fraction,
    mutation_rate,
    max_pop_size,
    accuracy_cutoff,
    learning_epochs,
):
    learning_epoch = 1
    population = initialize_population()
    length = get_data_length(data)
    birth_iteration = 1
    while learning_epoch < learning_epochs:
        for i in range(1, length):
            instance = get_instance(data, i)
            match_set = create_match_set(population, instance)
            correct_set = create_correct_set(match_set, instance)
            if len(correct_set) == 0:  # Activate covering if the correct set is empty
                # Create a new classifier that matches the current training instance
                classifier = covering(instance, birth_iteration, covering_specificity)
                population.append(classifier)  # add new classifier to the population
                birth_iteration += 1
            else:
                set_subsumption(correct_set)
                # If the correct set is not empty, activate the genetic algorithm
                genetic_algorithm(
                    population,
                    correct_set,
                    tournament_size_fraction,
                    mutation_rate,
                    instance,
                    birth_iteration,
                )
                birth_iteration += 1
        # population_subsumption(population)
        deletion(population, max_pop_size)
        learning_epoch += 1
    compaction(population, accuracy_cutoff)
    return population


def testing2(rule_set, data):
    if len(rule_set) == 0:
        return None
    length = get_data_length(data)
    number_correct = 0
    for i in range(1, length):
        instance = get_instance(data, i)
        match_set = create_match_set(rule_set, instance)
        vote1 = sum(j["numerosity"] for j in match_set if j["action"] == 0)
        vote2 = sum(k["numerosity"] for k in match_set if k["action"] == 1)
        if vote1 > vote2:
            vote = 0
        if vote2 > vote1:
            vote = 1
        if vote == instance[-1]:
            number_correct += 1
    percent_correct = number_correct / (length - 1)
    return percent_correct


def main():
    population = initialize_population()
    print(population)
    data = Path(__file__).parents[2].joinpath("data/6Multiplexer_Data_Complete.csv")
    instance = get_instance(data, 1)
    print(instance)
    match_set = create_match_set(population, instance)
    print(population)
    print(match_set)
    x = [(0, 0), (1, 0), (5, 1), (4, 1)]
    y = [0, 0, 1, 1, 1, 1]
    print(does_match(x, y))
    correct_set = create_correct_set(match_set, instance)
    print(population)
    print(correct_set)
    classifier = covering(instance, 1, specificity=0.5)
    update_population(classifier, population)
    print(population)
    population = testing1(data, 0.5)
    print(population)
    print(len(population))
    tournament1 = tournament_selection(population, round(len(population) / 5))
    print(tournament1)
    print(len(tournament1))
    parent1 = parent_selection(tournament1)
    print(parent1)
    tournament2 = tournament_selection(population, round(len(population) / 5))
    print(tournament2)
    print(len(tournament2))
    parent2 = parent_selection(tournament2)
    print(parent1)
    print(parent2)
    son, daughter = crossover(parent1, parent2, 1)

    print(parent1)
    print(parent2)
    print(son)
    print(daughter)
    mutated = mutation(daughter, instance, 0.5)

    print(mutated)
    x = {
        "state": [
            (0, 0),
            (1, 1),
            (4, 1),
        ],
        "action": 1,
        "numerosity": 1,
        "match count": 1,
        "correct count": 1,
        "accuracy": 1,
        "fitness": 1,
        "deletion vote": 1,
        "birth iteration": 1,
    }
    y = {
        "state": [(0, 0), (1, 1), (4, 1)],
        "action": 1,
        "numerosity": 1,
        "match count": 1,
        "correct count": 1,
        "accuracy": 1,
        "fitness": 1,
        "deletion vote": 1,
        "birth iteration": 1,
    }
    print(more_general(x, y))

    rule_set = binaryLCS(data, 0.5, 0.5, 0.3, 1000, 1.0, 300)
    pprint.pprint(rule_set)


if __name__ == "__main__":
    main()
