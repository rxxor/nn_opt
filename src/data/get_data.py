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
import logging

# =============================================================================
# Chapter 1: Set Logging Parametes
# =============================================================================
logging.basicConfig(filename='reports/logs/run_status.txt',
                    format='%(asctime)s-%(levelname)s:%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

# =============================================================================
# Chapter 2: Import Data
# =============================================================================
logging.info('Loading predictor variables')
X = load_boston().data

logging.info('Loading target variables')
y = load_boston().target

logging.info('Loading predictor names')
cols = load_boston().feature_names
logging.debug(cols)

# =============================================================================
# Chapter 3: Save Data
# =============================================================================
logging.info('Saving data to csv')
np.savetxt('data/raw/predictors.csv', X, delimiter=',')
np.savetxt('data/raw/target.csv', y, delimiter=',')
with open('data/raw/features.txt', 'w') as f:
    f.write(','.join(cols))
