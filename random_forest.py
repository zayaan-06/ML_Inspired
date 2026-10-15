#!/usr/bin/python3
import utils

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting Random Forest...')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=30, max_features=50)
clf.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, clf.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, clf.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(clf, 'new_models/rf_subset_feature_50.joblib')
