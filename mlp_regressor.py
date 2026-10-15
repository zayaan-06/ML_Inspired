#!/usr/bin/python3
import utils

name = 'mlp_regressor_20_10_5_3'

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting ' + name + '...')
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(20,10,5,3), alpha=0.008, learning_rate='adaptive', max_iter=1000)
model.fit(train_x, train_y)

print('\nResults on train:', model.score(train_x, train_y))

print('\nResults on test:', model.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(model, 'annual_regression_models/' + name + '.joblib')
