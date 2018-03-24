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
            X.append(list(map(float, d)))

    return X, y

def min_max_scaling(d):
    pass

def normalize(data):
    scaleddata = []
    for d in data:
        for col in d:
            ds = min_max_scaling(d)
            scaleddata.append(ds)

def normalize_np(data):
    minn = data.min()
    maxx = data.max()
    range = maxx - minn
    for d in np.nditer(data):
        val = 2 * ((d - minn) / range) - 1
        d = val

def svm_classify(X, y, deg=1):
    prob = svm_problem(X, y)
    for i in range(deg):
        param = svm_parameter("-t 1 -d {0}".format(i))
        svm_train(prob, param)


if __name__ == '__main__':
    X, y = getdata()
    stringcol = X[:, 0]
    X = np.delete(X, 0, 1)
    X = X.astype(np.float)
    for i in range(X.shape[1]):
        r = X[:, i]
        normalize_np(r)

    svm_classify(X, y)
