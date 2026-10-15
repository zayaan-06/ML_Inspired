#!/usr/bin/python3
import utils
from sklearn.naive_bayes import GaussianNB

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting Gaussian NB model...')
model = GaussianNB()
model.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, model.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, model.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(model, 'new_models/gaussian_nb.joblib')
