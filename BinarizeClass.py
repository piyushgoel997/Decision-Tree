from collections import Counter

import numpy as np


def binarize(classes):
    mapping = {}
    b = True
    for cls, _ in sorted(dict(Counter(classes)).items(), key=lambda i: i[1]):
        if b:
            mapping[cls] = 0
        else:
            mapping[cls] = 1
        b = not b
    return np.array([mapping[cls] for cls in classes])


d = np.load("car.npy", allow_pickle=True)
d[:, -1] = binarize(d[:, -1])
print(sum(d[:, -1]), d.shape[0] - sum(d[:, -1]))
np.save("DataSets/Bin_classes/car.npy", d)
