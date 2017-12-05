#! python2
# coding: utf-8
import numpy as np
import pandas as pd
import readData as rd
import math


def rmse(m1, m2):
    return np.sqrt(np.mean((m1 - m2) ** 2))


def getRMSE():
    trainMatrix = rd.loadTrainMatrix()
    testMatrix = rd.loadTestMatrix()
    simMatrix = rd.loadSimMatrix()
    trainMatrixOne = np.ones(trainMatrix.shape, dtype='float32')
    trainMatrixOne[trainMatrix == 0] = 0
    rfMatrix = np.dot(simMatrix, trainMatrix) / \
        np.dot(simMatrix, trainMatrixOne)
    rfMatrix = np.nan_to_num(rfMatrix)
    testMatrixOne = np.ones(testMatrix.shape, dtype='float32')
    testMatrixOne[testMatrix == 0] = 0
    print 'RMSE: %s' % rmse(rfMatrix * testMatrixOne, trainMatrix)


def main():
    getRMSE()


if __name__ == '__main__':
    main()
