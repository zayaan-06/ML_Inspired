#!/usr/bin/python3
import utils
from sklearn.linear_model import LogisticRegression

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting logistic regression model...')
log_reg = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
log_reg.fit(train_x, train_y)

print('\nResults on train:')
utils.print_report(train_y, log_reg.predict(train_x))

print('\nResults on test:')
utils.print_report(test_y, log_reg.predict(test_x))

print('\nDumping model')
from joblib import dump
dump(log_reg, 'new_models/log_reg.joblib')
