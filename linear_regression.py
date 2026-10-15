#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(False)

print('Fitting LinearRegression...')
from sklearn.linear_model import LinearRegression
lr = LinearRegression(n_jobs=-1)
lr.fit(train_x, train_y)

print('\nResults on train:', lr.score(train_x, train_y))

print('\nResults on test:', lr.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(lr, 'annual_regression_models/linear_regression.joblib')

