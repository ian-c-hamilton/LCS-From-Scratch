import random


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
