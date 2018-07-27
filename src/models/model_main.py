# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:53:10 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import modules
# =============================================================================
import numpy as np
import logging.config
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sys.path.append(os.getcwd())

from src.models.artemis import Artemis

# =============================================================================
# Chapter 1: Set Logging Parameters
# =============================================================================
logging.config.fileConfig('logging.ini')


# =============================================================================
# Chapter 2: Import data
# =============================================================================
def load_data():
    logging.info('Load data')
    X = np.loadtxt('data/raw/predictors.csv', delimiter=',')
    y = np.loadtxt('data/raw/target.csv')
    with open('data/raw/features.txt', 'r') as f:
        cols = f.read().split(',')

    return X, y, cols


if __name__ == '__main__':
    X, y, cols = load_data()
    test = Artemis(X.shape[1])
    test.train(X, y)
    y_hat = test.predict(X)
    plt.scatter(y, y_hat)
    plt.show()
