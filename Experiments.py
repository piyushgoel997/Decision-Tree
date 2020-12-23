import random
import time
from multiprocessing import Pool

import numpy as np

from Sampling_techniques import select_random_feature, select_lease_expected_uncertainty_feature, \
    select_heuristic_feature
from Tree import make_prediction, predict_cls


class Experiment:

    def __init__(self, tree, feature_cols, remove_ratio, features):
        self.tree = tree
        self.feature_cols = feature_cols
        self.remove_ratio = remove_ratio
        self.features = features

    def single_experiment(self, t):
        (x, y) = t
        correct = [0, 0, 0, 0]
        sampling_times = [0, 0, 0, 0]

        temp = np.copy(x)
        for i in range(len(self.feature_cols) - 1):
            if random.random() < self.remove_ratio:
                for i_ in range(self.feature_cols[i], self.feature_cols[i + 1]):
                    temp[i_] = -1
        # print(temp)

        # no sampling
        if predict_cls(make_prediction(self.tree, temp)) == y: correct[3] += 1

        # first unknown feature sampling
        s_time = time.time()
        heu = np.copy(temp)
        i = select_heuristic_feature(self.tree, temp)
        heu[i] = x[i]
        if predict_cls(make_prediction(self.tree, heu)) == y: correct[0] += 1
        sampling_times[0] += time.time() - s_time

        # least expected uncertainty sampling
        s_time = time.time()
        leu = np.copy(temp)
        i = select_lease_expected_uncertainty_feature(self.tree, leu, self.features)
        leu[i] = x[i]
        if predict_cls(make_prediction(self.tree, leu)) == y: correct[1] += 1
        sampling_times[1] += time.time() - s_time

        # random sampling
        s_time = time.time()
        rf = np.copy(temp)
        i = select_random_feature(rf)
        rf[i] = x[i]
        if predict_cls(make_prediction(self.tree, rf)) == y: correct[2] += 1
        sampling_times[2] += time.time() - s_time

        return np.array(correct), np.array(sampling_times)

    def run_exp(self, X_test, Y_test):
        correct = np.zeros((4,))
        sampling_times = np.zeros((4,))
        pool = Pool()
        res = pool.map(self.single_experiment, zip(X_test, Y_test))
        for c, st in res:
            correct += c
            sampling_times += st
        correct /= len(Y_test)
        sampling_times /= len(Y_test)
        print("Accuracies ->", "No sampling =", correct[3], "Random selection =", correct[2], "Heuristic selection =",
              correct[0], "Least Expected Uncertainty =", correct[1])
        print("Avg sampling time ->Random selection =", sampling_times[2], "Heuristic selection =", sampling_times[0],
              "Least Expected Uncertainty =", sampling_times[1])
        return correct, sampling_times