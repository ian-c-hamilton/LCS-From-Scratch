from lcs import utils


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
            if utils.does_match(state, instance) == True:
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
