#!/usr/bin/python3
import utils

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(True)

print('Fitting LinearSVC...')
from sklearn.svm import LinearSVC
svc = LinearSVC(class_weight='balanced', dual=False, penalty='l2', loss='squared_hinge')
svc.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, svc.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, svc.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(svc, 'new_models/svc_l2_squared_hinge.joblib')

