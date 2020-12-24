import random

import numpy as np

from Tree import make_prediction2


def select_heuristic_feature(tree, x):
    if tree.leaf:
        return -1
    if x[tree.feature] == -1:
        return tree.feature
    if x[tree.feature] in tree.children:
        return select_heuristic_feature(tree.children[x[tree.feature]], x)
    for c in tree.children.values():
        return select_heuristic_feature(c, x)


def calc_uncertainty(predictions):
    return 1 - max(predictions.values())


def cal_uncertainty2(predictions):
    u = 1
    for p in predictions:
        u *= p
    return u


def select_lease_expected_uncertainty_feature(tree, incomplete_x, features):
    min_expected_uncertainty = 1
    idx = -1
    for i, v in enumerate(incomplete_x):
        if v == -1:
            tc = 0
            expected_uncertainty = 0
            for f, c in features[i].items():
                x = np.copy(incomplete_x)
                x[i] = f
                expected_uncertainty += c * calc_uncertainty(make_prediction2(tree, x))
                tc += c
            expected_uncertainty /= tc
            if expected_uncertainty <= min_expected_uncertainty:
                min_expected_uncertainty = expected_uncertainty
                idx = i
    return idx


def select_random_feature(x):
    unk_features = []
    for i, v in enumerate(x):
        if v == -1:
            unk_features.append(i)
    if len(unk_features) == 0:
        return 0
    return random.choice(unk_features)
