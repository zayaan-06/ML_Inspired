#!/usr/bin/python3
import utils

name = 'random_forest_regressor'

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting ' + name + '...')
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, min_samples_split=0.01)
model.fit(train_x, train_y)

print('\nResults on train:', model.score(train_x, train_y))

print('\nResults on test:', model.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(model, 'regression_models/' + name + '.joblib')
