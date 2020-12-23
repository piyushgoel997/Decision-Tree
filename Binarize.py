import sys
from collections import Counter

import numpy as np
import pandas as pd


def binarize_feature(feature, divs=-1):
    if divs > 0:
        try:
            feature_ = pd.cut(feature, bins=divs)
            feature = feature_
        except:
            # do nothing
            print("except")
    if divs == 2:
        # make 0-1
        attrs = np.zeros((len(feature), 1))
        for i in range(len(feature)):
            if feature[i] == feature[0]: attrs[i, 0] = 1
        print(np.sum(attrs), len(feature) - np.sum(attrs))
    else:
        # make one-hot
        vals = list(set(feature))
        attrs = np.zeros((len(feature), len(vals)))
        for i in range(len(feature)):
            attrs[i, vals.index(feature[i])] = 1
        print(np.sum(attrs, axis=0))
    return attrs


def binarize_classes(classes):
    mapping = {}
    b = True
    for cls, _ in sorted(dict(Counter(classes)).items(), key=lambda i: i[1]):
        if b:
            mapping[cls] = 0
        else:
            mapping[cls] = 1
        b = not b
    result = np.array([mapping[cls] for cls in classes])
    result = result.reshape((len(result), 1))
    print(sum(result), len(result) - sum(result))
    return result


name = "faults"
file_name = "DataSets/New folder/New Folder/" + name + ".csv"
sys.stdout = open("DataSets/" + name + "_log.txt", 'w')
f = pd.read_csv(file_name, header=None, delimiter=',')
cols = f.columns
features = []
f_c = [0]
split_at = -1
for c in cols[:split_at]:
    num_feature_values = len(set(f[c]))
    print(num_feature_values)
    if num_feature_values > 20:
        features.append(binarize_feature(f[c], divs=10))
    elif num_feature_values == 2:
        features.append(binarize_feature(f[c], divs=2))
    elif num_feature_values == 1:
        continue
    else:
        features.append(binarize_feature(f[c]))
    f_c.append(f_c[-1] + features[-1].shape[1])
# features.append(np.array(f[cols[-1]]).reshape((len(f[cols[-1]]), 1)))
for i, c in enumerate(cols[split_at:]):
    data = features.copy()
    data.append(binarize_classes(f[c]))
    data = np.concatenate(data, axis=1)
    print(i)
    print(data.shape)
    n = name + "_" + str(i)
    # else: n = name
    np.save("DataSets/" + n + ".npy", data)
    np.save("DataSets/" + n + "_fc.npy", f_c)
# features.append(binarize_feature(f[cols[-1]], divs=2))
# data = np.concatenate(features, axis=1)
# print(data.shape)
# np.save("DataSets/" + name + ".npy", data)
# np.save("DataSets/" + name + "_fc.npy", f_c)
