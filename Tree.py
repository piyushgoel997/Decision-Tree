MIN_ELEMENTS = 1


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


def make_tree(X, Y, features_left, indices_left, measure_impurity=gini):
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
        return make_prediction(tree.children[x[tree.feature]], x)
    results = {}
    for child in tree.children.values():
        p = make_prediction(child, x)
        for k in p.keys():
            if k not in results:
                results[k] = 0
            results[k] = results[k] + (child.count / tree.count) * p[k]
    return results


def make_prediction2(tree, x):
    if tree.leaf:
        return tree.feature
    if x[tree.feature] in tree.children:
        res = make_prediction(tree.children[x[tree.feature]], x)
        for k in res.keys():
            res[k] = res[k] * (tree.children[x[tree.feature]].count / tree.count)
        return res
    results = {}
    for child in tree.children.values():
        p = make_prediction(child, x)
        for k in p.keys():
            if k not in results:
                results[k] = 0
            results[k] = results[k] + (child.count / tree.count) * p[k]
    return results


def predict_cls(pred):
    max_prob = 0
    cls = -1
    for k in pred.keys():
        if pred[k] > max_prob:
            max_prob = pred[k]
            cls = k
    return cls
