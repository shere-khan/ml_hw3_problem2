from svmutil import *
import numpy as np


# y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
# y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
# prob  = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 4 -b 1')
# m = svm_train(prob, param)

def getdata():
    fn = 'yeast.data'
    with open(fn) as f:
        X = []
        y = []
        x1 = []
        for line in f:
            d = line.split()
            x1.append(d.pop(0))
            y.append(d.pop(-1))
            x = list(map(float, d))
            for d in x:
                if d < 0:
                    print('test1')
            X.append(x)

    return (x1, X), y

def min_max_scaling(d, minn, r):
    return 2 * ((d - minn) / r) - 1

def normalize(X, mins, maxs):
    scaleddata = []
    for x in X:
        ex = []
        for i in range(len(x)):
            r = maxs[i] - mins[i]
            ex.append(min_max_scaling(x[i], mins[i], r))
        scaleddata.append(ex)

    return scaleddata

def normalize_np(data):
    minn = data.min()
    maxx = data.max()
    range = maxx - minn
    for d in np.nditer(data):
        d = min_max_scaling(d, minn, range)

def svm_classify(X, y, deg=1):
    prob = svm_problem(X, y)
    for i in range(deg):
        param = svm_parameter("-t 1 -d {0}".format(i))
        svm_train(prob, param)

def find_min_max(X):
    mins = list(map(int, "0 0 0 0 0 0 0 0".split()))
    maxs = list(map(int, "0 0 0 0 0 0 0 0".split()))
    for x in X:
        for i in range(len(x)):
            mins[i] = min(x[i], mins[i])
            maxs[i] = max(x[i], maxs[i])

    return mins, maxs


if __name__ == '__main__':
    (x1, X), y = getdata()
    mins, maxs = find_min_max(X)
    X = normalize(X, mins, maxs)
    print('dkfj')
    # stringcol = X[:, 0]
    # X = np.delete(X, 0, 1)
    # X = X.astype(np.float)
    # for i in range(X.shape[1]):
    #     r = X[:, i]
    #     normalize_np(r)

    # svm_classify(X, y)
