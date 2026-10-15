#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

from joblib import load
model = load('new_models/mlp.joblib')

import numpy as np
train_predict = model.predict_proba(train_x)[:,1]

ordered_profit_ratios = train_y[(-train_predict).argsort()] - 1.0
print(ordered_profit_ratios)
n = len(ordered_profit_ratios)
invested_amount = np.array([25 * top_k for top_k in range(1, n+1)])
net_profit = np.array([np.sum(ordered_profit_ratios[:top_k]) / top_k for top_k in range(1, n+1)])
print(invested_amount)
print(net_profit)

'''
for i in range(len(net_profit)):
    if i > 1500 and net_profit[i] < 0.375:
        print("Cutoff =", i)
        break

cutoff = -np.sort(-train_predict)[i]

'''
import matplotlib.pyplot as plt
plt.plot(invested_amount, net_profit)
plt.savefig('classification_eval.png')

'''
print('\nResults on test:', model.score(test_x, test_y))
invested_amount = 0
profit = 0
test_predict = model.predict(test_x)
for i in range(len(test_predict)):
    pred = test_predict[i]
    if pred >= cutoff:
        invested_amount += 25
        profit += 25 * (test_y[i] - 1.0)
print("invested_amount =", invested_amount)
print("profit =", profit)
'''
