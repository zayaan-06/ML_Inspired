#!/usr/bin/python3
import utils

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting MLP...')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(10, 10, 5, 3))
mlp.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, mlp.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, mlp.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(mlp, 'new_models/mlp.joblib')
