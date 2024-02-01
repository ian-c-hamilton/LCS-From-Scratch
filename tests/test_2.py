from lcs import utils
from lcs.set_creation import create_match_set


def testing2(rule_set, data):
    if len(rule_set) == 0:
        return None
    length = utils.get_data_length(data)
    number_correct = 0
    for i in range(1, length):
        instance = utils.get_instance(data, i)
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
