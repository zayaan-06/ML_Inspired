#!/usr/bin/python3
import utils

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data()

print('Fitting SGDClassifier...')
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(class_weight='balanced', loss='squared_hinge', tol=1e-5, max_iter=10000)
sgd.fit(train_x, train_y)

utils.print_report(test_y, sgd.predict(test_x))
