#! python2
# coding: utf-8
import numpy as np
import readData as rd
import time


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print "%s cost %s second" % (fn.__name__, time.clock() - start)
    return _wrapper


def rmse(m1, m2, l):
    one = np.ones(m2.shape, dtype='float32')
    one[m2 == 0] = 0
    return np.sqrt(np.sum((m1 * one - m2) ** 2) / l)


@time_me
def getRMSE(trainMatrix, testMatrix, simMatrix, l):
    trainMatrixOne = np.ones(trainMatrix.shape, dtype='float32')
    trainMatrixOne[trainMatrix == 0] = 0
    rfMatrix = np.dot(simMatrix, trainMatrix) / \
        np.dot(simMatrix, trainMatrixOne)
    rfMatrix = np.nan_to_num(rfMatrix)
    print 'RMSE: %s' % rmse(rfMatrix, testMatrix, l)


def main():
    trainMatrix = rd.loadTrainMatrix()
    testMatrix = rd.loadTestMatrix()
    simMatrix = rd.loadSimMatrix()
    l = rd.getTestLen()
    getRMSE(trainMatrix, testMatrix, simMatrix, l)


if __name__ == '__main__':
    main()
