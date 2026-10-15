#!/usr/bin/python3
import utils

name = 'random_forest_shallow_regressor_10'

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

print('Fitting ' + name + '...')
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
model.fit(train_x, train_y)

print('\nResults on train:', model.score(train_x, train_y))

print('\nResults on test:', model.score(test_x, test_y))

print('\nDumping model')
from joblib import dump
dump(model, 'annual_regression_models/' + name + '.joblib')
