#!/usr/bin/python3
import utils

name = 'linear_svr'

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting ' + name + '...')
from sklearn.svm import LinearSVR
model = LinearSVR(loss='squared_epsilon_insensitive', dual=False)
model.fit(train_x, train_y)

print('\nResults on train:', model.score(train_x, train_y))

print('\nResults on test:', model.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(model, 'regression_models/' + name + '.joblib')

