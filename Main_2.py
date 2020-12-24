import argparse
import sys
import time
from collections import Counter

import numpy as np

from Experiments import predict_cls, Experiment
from Tree import make_prediction, make_tree

MIN_ELEMENTS = 1
NUM_EXPERIMENTS = 10


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
    data = np.load("data/" + args.data + ".npy", allow_pickle=True)
    feature_cols = np.load("data/" + args.data + "_fc.npy")
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
        tree = make_tree(X_train, Y_train, list(range(X_train.shape[1])), list(range(X_train.shape[0])))
        print("tree made in", time.time() - exp_start_time)
        ca = complete_data_test(X_test, Y_test, tree)
        print("Complete data acc =", ca)
        complete_accuracy += ca
        for j, r in enumerate(remove_ratios):
            e = Experiment(tree, feature_cols, r, features)
            a, s = e.run_exp(X_test, Y_test)
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
    print("Accuracy of trivial classifier = {:.5f}".format(
        max(sum(data[:, -1])/len(data[:, -1]), (len(data[:, -1]) - sum(data[:, -1]))/len(data[:, -1]))))
    print("total time taken", time.time() - start_time)
    print("Class counts in the data", Counter(list(data[:, -1])))
    print("Number of total instances =", data.shape[0], "\nNumber of attributes =", data.shape[1])