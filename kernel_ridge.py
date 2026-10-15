#!/usr/bin/python3
import utils

name = 'kernel_ridge'

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting ' + name + '...')
from sklearn.kernel_ridge import KernelRidge 
model = KernelRidge()
model.fit(train_x, train_y)

print('\nResults on train:', model.score(train_x, train_y))

print('\nResults on test:', model.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(model, 'regression_models/' + name + '.joblib')

