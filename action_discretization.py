from itertools import combinations
from scipy.special import comb
import numpy as np

#This function is to build action space by combination
def Action_discretization(asset_num, division):

    item_num = asset_num + division - 1
    action_num = int(comb(item_num, asset_num - 1))
    actions = {}
    pointer = 0

    for c in combinations(np.arange(item_num), asset_num - 1):
        action = np.zeros(asset_num)
        for i in range(len(c) - 1):
            action[i + 1] = c[i + 1] - c[i] - 1
        action[0] = c[0]
        action[-1] = item_num - c[-1] - 1
        actions[pointer] = action / division
        pointer += 1

    return action_num, actions


