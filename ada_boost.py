#!/usr/bin/python3
import utils

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting Ada Boost...')
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=200)
clf.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, clf.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, clf.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(clf, 'new_models/adaboost_200.joblib')
