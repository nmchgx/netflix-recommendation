#! python2
# coding: utf-8
import numpy as np
import pandas as pd
from tqdm import tqdm


def saveUserKey():
    data = np.loadtxt('data/users.txt', dtype='int')
    np.save('output/user_key.npy', np.array([data, range(0, len(data))]).T)


def loadUserKey():
    return dict(np.load('data/user_key.npy'))


def initMatrix(data, name):
    users = np.load('data/user_key.npy')
    movies = pd.read_table('data/movie_titles.txt', header=None,
                           delim_whitespace=True, names=['movie_id', 'year', 'title'])
    usersDict = dict(users)
    users_len = len(users)
    movie_len = len(movies)
    matrix = np.zeros((users_len, movie_len), dtype='float32')
    for item in tqdm(data):
        matrix[usersDict[item[0]]][item[1] - 1] = item[2]
    np.save(name, matrix)


def saveTrainMatrix():
    df = pd.read_table('data/netflix_train.txt', header=None, delim_whitespace=True,
                       names=['user_id', 'movie_id', 'score', 'time'])
    data = np.array(df[['user_id', 'movie_id', 'score']].values, dtype='int')
    initMatrix(data, 'output/matrix_train_new.npy')


def loadTrainMatrix():
    return np.load('data/matrix_train.npy')


def saveTestMatrix():
    df = pd.read_table('data/netflix_test.txt', header=None, delim_whitespace=True,
                       names=['user_id', 'movie_id', 'score', 'time'])
    data = np.array(df[['user_id', 'movie_id', 'score']].values, dtype='int')
    initMatrix(data, 'output/matrix_test_new.npy')


def loadTestMatrix():
    return np.load('data/matrix_test.npy')


def saveSimMatrix():
    data = np.load('data/matrix_train.npy')
    data2 = np.dot(data, data.T)
    matrix = np.dot(data, data.T) / np.dot(np.linalg.norm(data, axis=1).reshape(len(data), 1),
                                           np.linalg.norm(data.T, axis=0).reshape(1, len(data)))
    np.save('output/matrix_sim_train.npy', matrix)


def loadSimMatrix():
    return np.load('data/matrix_sim_train.npy')
