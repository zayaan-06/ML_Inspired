#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)

from joblib import load
model = load('annual_regression_models/random_forest_shallow_regressor_10.joblib')

import numpy as np
print('\nScore on train:', model.score(train_x, train_y))
train_predict = model.predict(train_x)

ordered_profit_ratios = train_y[(-train_predict).argsort()] - 1.0
print(ordered_profit_ratios)
n = len(ordered_profit_ratios)
invested_amount = np.array([top_k / len(train_x) for top_k in range(100, n+1)])
net_profit = np.array([np.sum(ordered_profit_ratios[:top_k]) / top_k for top_k in range(100, n+1)])
print(invested_amount)
print(net_profit)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
with plt.style.context('Solarize_Light2'):
    plt.plot(invested_amount * 100, net_profit * 100)
    plt.title('Investment Return Trend')
    plt.xlabel('percentage of loans invested')
    plt.ylabel('annualized return rate')
    fmt = mtick.FormatStrFormatter('%.0f%%')
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gca().yaxis.set_major_formatter(fmt)
plt.savefig('regression_annual_eval_train.svg')

for i in range(len(net_profit)):
    if net_profit[i] < 0.156:
        print("Cutoff =", i)
        break

cutoff = -np.sort(-train_predict)[i]
print('Real cutoff =', cutoff)


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

test_predict = model.predict(test_x)

ordered_profit_ratios = test_y[(-test_predict).argsort()] - 1.0
print(ordered_profit_ratios)
n = len(ordered_profit_ratios)
invested_amount = np.array([top_k / len(test_x) for top_k in range(100, n+1)])
net_profit = np.array([np.sum(ordered_profit_ratios[:top_k]) / top_k for top_k in range(100, n+1)])
print(invested_amount)
print(net_profit)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
with plt.style.context('Solarize_Light2'):
    plt.plot(invested_amount * 100, net_profit * 100)
    plt.title('Investment Return Trend')
    plt.xlabel('percentage of loans invested')
    plt.ylabel('annualized return rate')
    fmt = mtick.FormatStrFormatter('%.0f%%')
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gca().yaxis.set_major_formatter(fmt)
plt.savefig('regression_annual_eval_test.svg')
'''
