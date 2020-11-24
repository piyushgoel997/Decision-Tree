# training data is full, but testing with missing features
import random
import time
from collections import Counter

import numpy as np

MIN_ELEMENTS = 1
NUM_EXPERIMENTS = 1


def gini(Y, ind):
    nums = {}
    for i in ind:
        y = Y[i]
        if y not in nums:
            nums[y] = 0
        nums[y] = nums[y] + 1
    g = 1
    for n in nums.values():
        g = g - (n / len(Y)) ** 2
    return g


class TreeNode:
    def __init__(self, feature, leaf, count):
        self.feature = feature
        self.leaf = leaf
        self.count = count
        self.children = {}


def make_tree(X, Y, features_left, indices_left, measure_impurity):
    # print(len(indices_left))
    if len(features_left) == 0 or len(indices_left) == MIN_ELEMENTS:
        counts = {}
        for i in indices_left:
            if Y[i] not in counts:
                counts[Y[i]] = 0
            counts[Y[i]] = counts[Y[i]] + 1
        # max_ct = 0
        # cls = -1
        # for k in counts.keys():
        #     if counts[k] > max_ct:
        #         max_ct = counts[k]
        #         cls = k
        return TreeNode(counts, True, len(indices_left))

    min_impurity = 1
    split_feature = -1
    saved_split = {}
    for f in features_left:
        ind = {}
        for i in indices_left:
            if X[i, f] not in ind:
                ind[X[i, f]] = []
            ind[X[i, f]].append(i)
        impurity = sum([len(i) * measure_impurity(Y, i) for i in ind.values()]) / len(indices_left)
        if impurity < min_impurity:
            min_impurity = impurity
            split_feature = f
            saved_split = ind

    node = TreeNode(split_feature, False, len(indices_left))
    for k in saved_split.keys():
        left = [*features_left]
        left.remove(split_feature)
        node.children[k] = make_tree(X, Y, left, saved_split[k], measure_impurity)

    return node


def make_prediction(tree, x):
    if tree.leaf:
        # return {tree.feature: 1}
        return tree.feature
    if x[tree.feature] in tree.children:
        return make_prediction(tree.children[x[tree.feature]], x)
    results = {}
    for child in tree.children.values():
        p = make_prediction(child, x)
        for k in p.keys():
            if k not in results:
                results[k] = 0
            results[k] = results[k] + (child.count / tree.count) * p[k]
    return results


temp_ct = 0


def select_heuristic_feature(tree, x):
    global temp_ct
    if tree.leaf:
        return -1
    if x[tree.feature] == -1:
        return tree.feature
    if x[tree.feature] in tree.children:
        return select_heuristic_feature(tree.children[x[tree.feature]], x)
    temp_ct += 1  # a higher value of this count is not good.
    for c in tree.children.values():
        return select_heuristic_feature(c, x)


def calc_uncertainty(predictions):
    # uncertainty = 1
    # for p in predictions.values():
    #     uncertainty *= p
    # return uncertainty
    return 1 - max(predictions.values())


def select_lease_expected_uncertainty_feature(tree, incomplete_x):
    min_expected_uncertainty = 1
    idx = -1
    for i, v in enumerate(incomplete_x):
        if v == -1:
            tc = 0
            expected_uncertainty = 0
            for f, c in features[i].items():
                x = np.copy(incomplete_x)
                x[i] = f
                expected_uncertainty += c * calc_uncertainty(make_prediction(tree, x))
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


# def make_prediction_ds_known(tree, incomplete_x, complete_x, known):
#     if tree.leaf:
#         # return {tree.feature: 1}
#         return tree.feature
#     x = np.copy(incomplete_x)
#     if x[tree.feature] == -1:
#         if known == 1:
#             x[tree.feature] = complete_x[tree.feature]
#         known -= 1
#     if x[tree.feature] in tree.children:
#         return make_prediction_ds_known(tree.children[x[tree.feature]], x, complete_x, known)
#     results = {}
#     for child in tree.children.values():
#         p = make_prediction_ds_known(child, x, complete_x, known)
#         for k in p.keys():
#             if k not in results:
#                 results[k] = 0
#             results[k] = results[k] + (child.count / tree.count) * p[k]
#     return results


# def print_accuracy(X_test, Y_test, tree, remove_features=0):
#     correct = np.zeros((remove_features + 1,))
#     for x, y in zip(X_test, Y_test):
#         temp = np.copy(x)
#         temp[random.sample(range(len(x)), remove_features)] = -1
#         for i in range(remove_features + 1):
#             pred = make_prediction(tree, temp, x, i)
#             # print(pred, y)
#             max_prob = 0
#             cls = -1
#             for k in pred.keys():
#                 if pred[k] > max_prob:
#                     max_prob = pred[k]
#                     cls = k
#             if cls == y:
#                 correct[i] += 1
#     print("features removed =", remove_features, "accuracy =", correct / len(X_test))


def predict_cls(pred):
    max_prob = 0
    cls = -1
    for k in pred.keys():
        if pred[k] > max_prob:
            max_prob = pred[k]
            cls = k
    return cls


def run_exp(X_test, Y_test, tree, remove_ratio=0.5):
    correct = np.zeros((3,))
    sampling_times = np.zeros((3,))
    j = 0
    for x, y in zip(X_test, Y_test):
        temp = np.copy(x)
        for i in range(len(x)):
            if random.random() < remove_ratio:
                temp[i] = -1
        print(j, "out of", len(Y_test))
        j += 1
        print(temp)

        s_time = time.time()
        heu = np.copy(temp)
        i = select_heuristic_feature(tree, temp)
        heu[i] = x[i]
        if predict_cls(make_prediction(tree, heu)) == y: correct[0] += 1
        sampling_times[0] += time.time() - s_time

        s_time = time.time()
        leu = np.copy(temp)
        i = select_lease_expected_uncertainty_feature(tree, leu)
        leu[i] = x[i]
        if predict_cls(make_prediction(tree, leu)) == y: correct[1] += 1
        sampling_times[1] += time.time() - s_time

        s_time = time.time()
        rf = np.copy(temp)
        i = select_random_feature(rf)
        rf[i] = x[i]
        if predict_cls(make_prediction(tree, rf)) == y: correct[2] += 1
        sampling_times[2] += time.time() - s_time
    correct /= len(Y_test)
    sampling_times /= len(Y_test)
    print("Accuracies ->Random selection =", correct[2], "Heuristic selection =", correct[0],
          "Least Expected Uncertainty =", correct[1])
    print("Avg sampling time ->Random selection =", sampling_times[2], "Heuristic selection =", sampling_times[0],
          "Least Expected Uncertainty =", sampling_times[1])
    return correct, sampling_times


# load data
data = np.load("census.npy", allow_pickle=True)
# np.random.shuffle(data)
# X_train = data[:int(0.8 * len(data)), :-1]
# Y_train = data[:int(0.8 * len(data)), -1]
#
# features = {}
# for i in range(X_train.shape[1]):
#     features[i] = Counter(X_train[:, i])
# # print(features)
# X_test = data[int(0.8 * len(data)):, :-1]
# Y_test = data[int(0.8 * len(data)):, -1]
#
# # make tree
# tree = make_tree(X_train, Y_train, list(range(X_train.shape[1])), list(range(X_train.shape[0])),
#                  gini)

remove_ratios = [0.25, 0.50, 0.75]
acc = np.zeros((len(remove_ratios), 3))
sampling_times = np.zeros((len(remove_ratios), 3))
start_time = time.time()
for i in range(NUM_EXPERIMENTS):
    exp_start_time = time.time()
    np.random.shuffle(data)
    X_train = data[:int(0.8 * len(data)), :-1]
    Y_train = data[:int(0.8 * len(data)), -1]

    features = {}
    for i in range(X_train.shape[1]):
        features[i] = Counter(X_train[:, i])
    print(features)
    X_test = data[int(0.8 * len(data)):, :-1]
    Y_test = data[int(0.8 * len(data)):, -1]

    # make tree
    print("making tree")
    tree = make_tree(X_train, Y_train, list(range(X_train.shape[1])), list(range(X_train.shape[0])),
                     gini)
    print("tree made in", time.time() - exp_start_time)
    for j, r in enumerate(remove_ratios):
        a, s = run_exp(X_test, Y_test, tree, remove_ratio=r)
        acc[j] += a
        sampling_times[j] += s
    print("Experiment done in", time.time() - exp_start_time)
for i in range(len(remove_ratios)):
    acc[i] /= NUM_EXPERIMENTS
# print(temp_ct)
for a, s, r in zip(acc, sampling_times, remove_ratios):
    print("==========================================================")
    print("Remove ratio =", r)
    print()
    print("Averaged Accuracies ->\nRandom selection =", a[2], "\nHeuristic selection =", a[0],
          "\nLeast Expected Uncertainty =", a[1])
    print()
    print("Averaged sampling times ->\nRandom selection =", s[2], "\nHeuristic selection =", s[0],
          "\nLeast Expected Uncertainty =", s[1])
    print("==========================================================")
print("total time taken", time.time() - start_time)

# for i in range(X_test.shape[1] + 1):
#     print_accuracy(X_test, Y_test, tree, remove_features=i)
# accuracy = 0.9335260115606936 features removed = 0
# accuracy = 0.8294797687861272 features removed = 1
# accuracy = 0.7630057803468208 features removed = 2
# accuracy = 0.7196531791907514 features removed = 3
# accuracy = 0.7167630057803468 features removed = 4
# accuracy = 0.7167630057803468 features removed = 5
# accuracy = 0.7167630057803468 features removed = 6

# features removed = 0 accuracy = [0.93352601]
# features removed = 1 accuracy = [0.8150289  0.93352601]
# features removed = 2 accuracy = [0.75433526 0.83815029 0.77456647]
# features removed = 3 accuracy = [0.71965318 0.80635838 0.75144509 0.73410405]
# features removed = 4 accuracy = [0.69942197 0.76300578 0.71098266 0.70809249 0.70520231]
# features removed = 5 accuracy = [0.6849711  0.70809249 0.69364162 0.68208092 0.6849711  0.6849711 ]
# features removed = 6 accuracy = [0.6849711 0.6849711 0.6849711 0.6849711 0.6849711 0.6849711 0.6849711]

# Averaged Accuracies ->
# Random selection = 0.7873988439306359
# Heuristic selection = 0.8106647398843928
# Least Expected Uncertainty = 0.8182658959537573
