from collections import Counter

import numpy as np
import pandas as pd


def categorize_feature(feature, divs):
    feature = pd.qcut(feature, q=divs, duplicates='drop')
    # vals = list(set(feature))
    # attrs = np.zeros((len(feature), 1))
    # for i in range(len(feature)):
    #     attrs[i, 0] = vals.index(feature[i])
    feature = np.array(feature).reshape((len(feature), 1))
    print(Counter(list(feature[:, 0])))
    return feature


file_name = "DataSets/New folder/student-por.csv"
out_name = "DataSets/student-por.npy"
f = pd.read_csv(file_name, delimiter=';')  # , header=None)
cols = f.columns
features = []
for c in cols[:-1]:
    num_feature_values = len(set(f[c]))
    print(num_feature_values)
    if num_feature_values > 20:
        features.append(categorize_feature(f[c], divs=10))
    else:
        features.append(np.array(f[c]).reshape((len(f[c]), 1)))
features.append(np.array(f[cols[-1]]).reshape((len(f[cols[-1]]), 1)))
data = np.concatenate(features, axis=1)
print(data.shape)
np.save(out_name, data)

# output_features = []
# for i in list(f.columns)[-4:-1]:
#     _f = f[i]
#     output_features.append(np.array(_f))
# data = np.concatenate(features, axis=1)
# print(data.shape)
# for i, o in enumerate(output_features):
#     o = o.reshape((len(o), 1))
#     print(o.shape)
#     np.save("DataSets/frogs" + str(i) + ".npy", np.concatenate((data, o), axis=1))
