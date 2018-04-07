from svmutil import *
import numpy as np
import random
import math as m
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def getdata():
    fn = 'yeast.data'
    with open(fn) as f:
        X = []
        y = []
        x1 = []
        for line in f:
            d = line.split()
            x1.append(d.pop(0))
            l = d.pop(-1)
            label = 1 if l == "CYT" else 0
            y.append(label)
            x = list(map(float, d))
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
        for x, lab in zip(X, y):
            f.write("{0} ".format(lab))
            for i, d in enumerate(x):
                if d != 0.0 or d != 0:
                    f.write("{0}:{1:.6f} ".format(i + 1, d))
            f.write("\n")

def sklprintvals(X, y):
    with open("sklearn_svc_results_k5_nosplit.txt", "w") as f:
        for d in range(1, 5):
            for c in range(1, 60):
                clf = SVC(C=c, kernel='poly', degree=d)
                S = cross_val_score(clf, np.array(X), np.array(y), cv=10)
                f.write("d: {0}  c: {1:>2}  error: {2:.4f}\n".format(d, c, 1 - S.mean()))
                print("d: {0}  c: {1:>2}  error: {2:.4f}".format(d, c, 1 - S.mean()))
            f.write("\n")
            print()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def cross_val_svm(X, y, k):
    chunks = chunkIt(list(zip(X, y)), k)
    for i in range(k):
        r = random.randint(len(chunks))
        chunk = chunks[r]
        trainy = [c[0] for c in chunk]
        trainx = [c[1] for c in chunk]

        m = svm_train(trainy, trainx, "-s 0 -t 1 -d 1")
        m = toPyModel(m)
        m.get_SV()
        # predict_y, predict_acc, predict_val = svm_predict(testy, testx, m)
        # accuracy, mse, scc = evaluations(testy, predict_y)
        # print("accuracy: {0}\nmean square error: {1}\nscc: {2}".format(accuracy, mse, scc))

if __name__ == '__main__':
    # Read data
    # (x1, X), y = getdata()

    # Create unscaled output file for grid.py
    # p = 0.9
    # capx = m.ceil(len(X) * p)
    # trainx = X[:capx]
    # testx = X[capx:]

    # capy = m.ceil(len(y) * p)
    # trainy = y[:capy]
    # testy = y[capy:]

    # Create unscaled training and test output file for easy.py
    # create_output(trainx, trainy, "training_unscaled.txt")
    # create_output(testx, testy, "test_unscaled.txt")


    # Normalize and create formatted output for libsvm python code
    # mins, maxs = find_min_max(X)
    # X = normalize(X, mins, maxs)
    # D = set_to_dict(set(y))
    # create_output(X, y, "training_and_test_binary.txt")

    # Scikit-learn cross validation attempt
    # sklprintvals(np.array(X), np.array(y))

    # Run libsvm
    y, X = svm_read_problem("training_and_test_binary.txt")

    # res = list(zip(y, X))
    # random.shuffle(res)
    # y = [d[0] for d in res]
    # X = [d[1] for d in res]
    # cross_val_svm(X, y, 10)

    # Create training and testing data
    p = 0.9
    capx = m.ceil(len(X) * p)
    trainx = X[:capx]
    testx = X[capx:]

    capy = m.ceil(len(y) * p)
    trainy = y[:capy]
    testy = y[capy:]

    m = svm_train(trainy, trainx, "-c 1 -g 1 -t 2 -d 3")
    predict_y, predict_acc, predict_val = svm_predict(testy, testx, m)
    accuracy, mse, scc = evaluations(testy, predict_y)
    print("accuracy: {0}\nmean square error: {1}\nscc: {2}".format(accuracy, mse, scc))
