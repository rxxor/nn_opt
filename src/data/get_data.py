# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:21:33 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import Modules
# =============================================================================
from sklearn.datasets import load_boston
import numpy as np
import logging.config

# =============================================================================
# Chapter 1: Set Logging Parametes
# =============================================================================
logging.config.fileConfig('logging.ini')


# =============================================================================
# Chapter 2: Import Data
# =============================================================================
def load_data():
    '''
    Function to load Boston Data from sklearn module
    '''
    logging.info('Loading predictor variables')
    X = load_boston().data

    logging.info('Loading target variables')
    y = load_boston().target

    logging.info('Loading predictor names')
    cols = load_boston().feature_names
    logging.debug(cols)

    return X, y, cols


# =============================================================================
# Chapter 3: Save Data
# =============================================================================
def save_data(X, y, cols):
    '''
    Function to save Boston Data to disk
    '''
    logging.info('Saving data to csv')
    np.savetxt('data/raw/predictors.csv', X, delimiter=',')
    np.savetxt('data/raw/target.csv', y, delimiter=',')
    with open('data/raw/features.txt', 'w') as f:
        f.write(','.join(cols))


if __name__ == '__main__':
    save_data(*load_data())
