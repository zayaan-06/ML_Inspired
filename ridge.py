#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting LinearRegression...')
from sklearn.linear_model import Ridge 
ridge = Ridge()
ridge.fit(train_x, train_y)

print('\nResults on train:', ridge.score(train_x, train_y))

print('\nResults on test:', ridge.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(ridge, 'annual_regression_models/ridge.joblib')

