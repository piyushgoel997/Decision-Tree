import numpy as np
import pandas as pd


def binarize_feature(feature, divs=-1):
    if divs == 2:
        # make 0-1
        vals = list(set(feature))
        attrs = np.zeros((len(feature), 1))
        for i in range(len(feature)):
            if feature[i] == vals[0]: attrs[i, 0] = 1
        print(np.sum(attrs), len(feature) - np.sum(attrs))
        return attrs

    if divs > 0:
        feature = pd.cut(feature, bins=divs)

    # make one-hot
    vals = list(set(feature))
    attrs = np.zeros((len(feature), len(vals)))
    for i in range(len(feature)):
        attrs[i, vals.index(feature[i])] = 1
    print(np.sum(attrs, axis=0))
    return attrs


##########################
# f = pd.read_csv("DataSets/abalone/abalone.data", header=None)
# features = [binarize_feature(f[0])]
# for i in range(1, 8):
#     _f = f[i]
#     if i == 3: features.append(binarize_feature(_f, divs=7))
#     else: features.append(binarize_feature(_f, divs=4))
# features.append(np.array(f[8]).reshape((len(f[8]), 1)))
# data = np.concatenate(features, axis=1)
# np.save("DataSets/abalone.npy", data)
###########################

##########################
# f = pd.read_csv("DataSets/Anuran Calls (MFCCs)/Frogs_MFCCs.csv")
# features = []
# for i in list(f.columns)[:-4]:
#     _f = f[i]
#     features.append(binarize_feature(_f, divs=5))
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
##########################

##########################
# f = pd.read_csv("DataSets/New folder/bank-full.csv", delimiter=';')
# cols = f.columns
# features = [binarize_feature(f[cols[0]], divs=4)]
# for i in range(1, 5):
#     features.append(binarize_feature(f[cols[i]]))
# features.append(binarize_feature(f[cols[5]], divs=4))
# for i in range(6, 9):
#     features.append(binarize_feature(f[cols[i]]))
# features.append(binarize_feature(f[cols[11]], divs=4))
# # 12 -> 15 incl
# for i in range(12, 15):
#     if i == 14: continue
#     features.append(binarize_feature(f[cols[i]], divs=4))
# features.append(binarize_feature(f[cols[15]]))
# features.append(np.array(f[cols[-1]]).reshape((len(f[cols[-1]]), 1)))
# data = np.concatenate(features, axis=1)
# print(data.shape)
# np.save("DataSets/bank.npy", data)
###########################

file_name = "DataSets/New folder/avila.txt"
out_name = "DataSets/avila.npy"
f = pd.read_csv(file_name)  # , delimiter=';')
cols = f.columns
features = []
for c in cols[:-1]:
    num_feature_values = len(set(f[c]))
    print(num_feature_values)
    if num_feature_values > 10:
        features.append(binarize_feature(f[c], divs=5))
    elif num_feature_values == 2:
        features.append(binarize_feature(f[c], divs=2))
    else:
        features.append(binarize_feature(f[c]))
features.append(np.array(f[cols[-1]]).reshape((len(f[cols[-1]]), 1)))
data = np.concatenate(features, axis=1)
print(data.shape)
np.save(out_name, data)
