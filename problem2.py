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
        for line in f:
            d = line.split()
            X = d[:-1]
            y = d[-1]

    return np.array(X), np.array(y)

def min_max_scaling(d):
    pass

def normalize(data):
    scaleddata = []
    for d in data:
        for col in d:
            ds = min_max_scaling(d)
            scaleddata.append(ds)

def normalize_np(data):
    for x in data.nditer(data.T):
        print(x)

if __name__ == '__main__':
    X, y = getdata()
    normalize_np(X[:, 1:9])
