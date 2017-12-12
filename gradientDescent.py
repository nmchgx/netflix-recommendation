#! python2
# coding: utf-8
import numpy as np
import readData as rd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor


def rmse(m1, m2, l):
    one = np.ones(m2.shape, dtype='float32')
    one[m2 == 0] = 0
    return np.sqrt(np.sum((m1 * one - m2) ** 2) / l)


def gradientDescent(X, k=50, lam=0.01, alpha=0.0001):
    m, n = X.shape
    U = np.random.random((m, k))
    V = np.random.random((n, k))
    U /= 10000
    V /= 10000
    A = np.ones(X.shape, dtype='float32')
    A[X == 0] = 0
    cycles = 1000
    lossArr = []
    rmseArr = []
    l = rd.getTestLen()
    now = int(time.time())
    print '============================================='
    print 'Gradient Descent cycles=%s k=%s lam=%s alpha=%s' % (cycles, k, lam, alpha)
    print '============================================='
    for step in xrange(cycles):
        tempU = alpha * (np.dot(A * (np.dot(U, V.T) - X), V) + 2 * lam * U)
        tempV = alpha * (np.dot((A * (np.dot(U, V.T) - X)).T, U) + 2 * lam * V)
        U -= tempU
        V -= tempV
        loss = 0.5 * (np.linalg.norm(A * (X - np.dot(U, V.T)), 'fro')**2) + \
            lam * (np.linalg.norm(U, 'fro')**2) + \
            lam * (np.linalg.norm(V, 'fro')**2)
        lossArr.append(loss)
        print 'step: %s | loss: %s' % (step + 1, loss)
        if (step + 1) % 10 == 0:
            r = rmse(np.dot(U, V.T), X, l)
            rmseArr.append(r)
            print '* step: %s | rmse: %s' % (step + 1, r)
            np.save('output/loss_arr_%s_%s_%s_%s_%s.npy' %
                    (cycles, k, lam, alpha, now), lossArr)
            np.save('output/rmse_arr_%s_%s_%s_%s_%s.npy' %
                    (cycles, k, lam, alpha, now), rmseArr)


def draw(x, y, label='loss', xLabel='generation', yLabel='loss', name='img/test.png', mode=0):
    sns.set(palette="muted", color_codes=True)
    plt.title('Convergence curve')
    plt.plot(x, y, label=label)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if mode == 0:
        plt.savefig(name)
        plt.close()


def gci(filePath, key):
    files = os.listdir(filePath)
    for f in files:
        if re.match(key, f):
            data = np.load(os.path.join(filePath, f))
            x = np.array(range(len(data)))
            if key == 'rmse':
                x *= 10
                print 'file=%s | RMSE=%s' % (f, data[-1])
            y = np.array(data)
            label = f[9:-15]
            name = 'img/' + f[:-3] + 'png'
            draw(x, y, label=label, yLabel=key, name=name, mode=1)
    plt.legend()
    plt.savefig('img/%s.png' % key)
    plt.close()


def main():
    trainMatrix = rd.loadTrainMatrix()
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(gradientDescent, trainMatrix, 50, 0.01, 0.0001)
        executor.submit(gradientDescent, trainMatrix, 20, 0.001, 0.0001)
        executor.submit(gradientDescent, trainMatrix, 50, 0.001, 0.0001)
        executor.submit(gradientDescent, trainMatrix, 20, 0.1, 0.0001)
        executor.submit(gradientDescent, trainMatrix, 50, 0.1, 0.0001)

    gci('output/output', 'loss')
    gci('output/output', 'rmse')


if __name__ == '__main__':
    main()
