#!/usr/bin/python3
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np

print('Preparing dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

print('Fitting logistic regression model...')
log_reg = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)

selector = RFE(log_reg, 900, step=1)
selector = selector.fit(train_x, train_y)
np.save('support.npy', selector.support_)
np.save('ranking.npy', selector.ranking_)
