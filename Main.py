import random

import numpy as np


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
        # self.not_present = None


def make_tree(X, Y, features_left, indices_left, measure_impurity):
    if len(features_left) == 0:
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


def print_accuracy(X_test, Y_test, tree, remove_features=0):
    correct = 0
    for x, y in zip(X_test, Y_test):
        temp = np.copy(x)
        temp[random.sample(range(len(x)), remove_features)] = -1
        pred = make_prediction(tree, temp)
        # print(pred, y)
        max_prob = 0
        cls = -1
        for k in pred.keys():
            if pred[k] > max_prob:
                max_prob = pred[k]
                cls = k
        if cls == y:
            correct += 1
    print("accuracy =", correct / len(X_test), "features removed =", remove_features)


# load data
data = np.load("car.npy", allow_pickle=True)
np.random.shuffle(data)
X_train = data[:int(0.8 * len(data)), :-1]
Y_train = data[:int(0.8 * len(data)), -1]

X_test = data[int(0.8 * len(data)):, :-1]
Y_test = data[int(0.8 * len(data)):, -1]

# make tree
tree = make_tree(X_train, Y_train, list(range(X_train.shape[1])), list(range(X_train.shape[0])),
                 gini)
for i in range(X_test.shape[1] + 1):
    print_accuracy(X_test, Y_test, tree, remove_features=i)
# accuracy = 0.9335260115606936 features removed = 0
# accuracy = 0.8294797687861272 features removed = 1
# accuracy = 0.7630057803468208 features removed = 2
# accuracy = 0.7196531791907514 features removed = 3
# accuracy = 0.7167630057803468 features removed = 4
# accuracy = 0.7167630057803468 features removed = 5
# accuracy = 0.7167630057803468 features removed = 6

