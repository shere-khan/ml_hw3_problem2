from svmutil import *
import numpy as np
import math as m
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


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

def svm_classify(deg=5):
    y, x = svm_read_problem('formatted_data.txt')
    prob = svm_problem(y, x)
    trains = []
    for i in range(1, deg + 1):
        param = svm_parameter("-t 1 -d {0} -v 10".format(i))
        m = svm_train(prob, param)
        trains.append(m)

    return trains

def find_min_max(X):
    mins = list(map(int, "0 0 0 0 0 0 0 0".split()))
    maxs = list(map(int, "0 0 0 0 0 0 0 0".split()))
    for x in X:
        for i in range(len(x)):
            mins[i] = min(x[i], mins[i])
            maxs[i] = max(x[i], maxs[i])

    return mins, maxs

def set_to_dict(S):
    D = {}
    for i, s in enumerate(S):
        D[s] = i + 1

    return D

def create_output(X, y, fn):
    with open(fn, "w") as f:
        for x, lab in zip(X,y):
            f.write("{0} ".format(lab))
            for i, d in enumerate(x):
                if d != 0.0 or d != 0:
                    f.write("{0}:{1:.6f} ".format(i + 1, d))
            f.write("\n")



if __name__ == '__main__':
    (x1, X), y = getdata()
    print()
    mins, maxs = find_min_max(X)
    X = normalize(X, mins, maxs)
    D = set_to_dict(set(y))
    y = list(map(lambda val: D[val], y))

    p = 0.8
    sx = m.ceil(len(X) * p)
    sy = m.ceil(len(y) * p)

    trainX = X[:sx]
    trainy = y[:sy]

    testX = X[sx:]
    testy = y[sy:]

    bestc = 0
    bestd = 0
    bests = 0
    for d in range(1, 5):
        # print("d: ", d)
        for c in range(1, 33):
            # print("c: ", c)
            clf = SVC(C=c, kernel='poly', degree=d)
            clf.fit(np.array(trainX), np.array(trainy))
            # S = clf.score(np.array(testX), np.array(testy), cv=10)
            S = cross_val_score(clf, np.array(testX), np.array(testy), cv=10)
            for s in S:
                if s > bests:
                    bests = s
                    bestc = c
                    bestd = d
                # print(s)
            print()
            print()

    print("best C: ", bestc)
    print("best d: ", bestd)

    # create_output(trainX, trainy, "training.txt")
    # create_output(testX, testy, "test.txt")

    # train = svm_classify()
    # print("results")
    # for t in train:
    #     print(t)
    # trains = svm_classify(X, y)
    # print()
    # stringcol = X[:, 0]
    # X = np.delete(X, 0, 1)
    # X = X.astype(np.float)
    # for i in range(X.shape[1]):
    #     r = X[:, i]
    #     normalize_np(r)

    # svm_classify(X, y)
