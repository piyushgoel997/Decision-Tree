import argparse
import random
import sys
import time
from collections import Counter

import numpy as np

MIN_ELEMENTS = 1
NUM_EXPERIMENTS = 10


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
    if len(features_left) == 0 or len(indices_left) == MIN_ELEMENTS:
        counts = {}
        for i in indices_left:
            if Y[i] not in counts:
                counts[Y[i]] = 0
            counts[Y[i]] = counts[Y[i]] + 1
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
        return tree.feature
    if x[tree.feature] in tree.children:
        # return make_prediction(tree.children[x[tree.feature]], x)
        res = make_prediction(tree.children[x[tree.feature]], x)
        for k in res.keys():
            res[k] = res[k] * (tree.children[x[tree.feature]].count/tree.count)
        return res
        ##
    results = {}
    for child in tree.children.values():
        p = make_prediction(child, x)
        for k in p.keys():
            if k not in results:
                results[k] = 0
            results[k] = results[k] + (child.count / tree.count) * p[k]
    return results


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


def predict_cls(pred):
    max_prob = 0
    cls = -1
    for k in pred.keys():
        if pred[k] > max_prob:
            max_prob = pred[k]
            cls = k
    return cls


def run_exp(X_test, Y_test, tree, feature_cols, remove_ratio=0.5):
    correct = np.zeros((4,))
    sampling_times = np.zeros((4,))
    j = 0
    for x, y in zip(X_test, Y_test):
        temp = np.copy(x)
        for i in range(len(feature_cols) - 1):
            if random.random() < remove_ratio:
                for i_ in range(feature_cols[i], feature_cols[i+1]):
                    temp[i_] = -1
        print(j, "out of", len(Y_test))
        j += 1
        print(temp)

        # no sampling
        if predict_cls(make_prediction(tree, temp)) == y: correct[3] += 1

        # first unknown feature sampling
        s_time = time.time()
        heu = np.copy(temp)
        i = select_heuristic_feature(tree, temp)
        heu[i] = x[i]
        if predict_cls(make_prediction(tree, heu)) == y: correct[0] += 1
        sampling_times[0] += time.time() - s_time

        # least expected uncertainty sampling
        s_time = time.time()
        leu = np.copy(temp)
        i = select_lease_expected_uncertainty_feature(tree, leu)
        leu[i] = x[i]
        if predict_cls(make_prediction(tree, leu)) == y: correct[1] += 1
        sampling_times[1] += time.time() - s_time

        # random sampling
        s_time = time.time()
        rf = np.copy(temp)
        i = select_random_feature(rf)
        rf[i] = x[i]
        if predict_cls(make_prediction(tree, rf)) == y: correct[2] += 1
        sampling_times[2] += time.time() - s_time
    correct /= len(Y_test)
    sampling_times /= len(Y_test)
    print("Accuracies ->", "No sampling =", correct[3], "Random selection =", correct[2], "Heuristic selection =",
          correct[0], "Least Expected Uncertainty =", correct[1])
    print("Avg sampling time ->Random selection =", sampling_times[2], "Heuristic selection =", sampling_times[0],
          "Least Expected Uncertainty =", sampling_times[1])
    return correct, sampling_times


def complete_data_test(X_test, Y_test, tree):
    correct = 0
    for x, y in zip(X_test, Y_test):
        if predict_cls(make_prediction(tree, x)) == y: correct += 1
    return correct / len(Y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()

    sys.stdout = open("logs/" + args.data + ".txt", "w")

    # load data
    data = np.load(args.data + ".npy", allow_pickle=True)
    feature_cols = np.load(args.data + "_fc.npy")
    remove_ratios = [0.25, 0.50, 0.75]
    acc = np.zeros((len(remove_ratios), 4))
    complete_accuracy = 0
    sampling_times = np.zeros((len(remove_ratios), 4))
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
        ca = complete_data_test(X_test, Y_test, tree)
        print("Complete data acc =", ca)
        complete_accuracy += ca
        for j, r in enumerate(remove_ratios):
            a, s = run_exp(X_test, Y_test, tree, feature_cols, remove_ratio=r)
            acc[j] += a
            sampling_times[j] += s
        print("Experiment done in", time.time() - exp_start_time)
    for i in range(len(remove_ratios)):
        acc[i] /= NUM_EXPERIMENTS
        sampling_times[i] /= NUM_EXPERIMENTS
    complete_accuracy /= NUM_EXPERIMENTS
    for a, s, r in zip(acc, sampling_times, remove_ratios):
        print("==========================================================")
        print("Remove ratio =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\nHeuristic selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[3], a[2], a[0], a[1]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Heuristic selection = {:.5f}\nLeast Expected Uncertainty = {:.5f}".format(s[3], s[2], s[0], s[1]))
        print("==========================================================")
    print("Accuracy with complete data = {:.5f}".format(complete_accuracy))
    print("Accuracy of trivial classifier =", max(sum(data[:, -1])/len(data[:, -1]),
                                                  (len(data[:, -1]) - sum(data[:, -1]))/len(data[:, -1])))
    print("total time taken", time.time() - start_time)
    print("Class counts in the data", Counter(list(data[:, -1])))
    print("Number of total instances =", data.shape[0], "\nNumber of attributes =", data.shape[1])