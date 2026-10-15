#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_data(load_existing=True)

from joblib import load
print('Loading models...')

joblibs = ['adaboost_200.joblib', 'log_reg.joblib', 'mlp.joblib', 'random_forest_200.joblib', 'svc_l2_squared_hinge.joblib']
models = []
for joblib in joblibs:
    models.append(load('new_models/' + joblib))

def voting_predict(x):
    import numpy as np
    scores = np.zeros(x.shape[0])
    for i in range(len(models)):
        print('Predictiing with model', joblibs[i])
        scores += models[i].predict(x)
    return (scores >= 3).astype(int)

def print_voting_results():
    print('\nResults on train:')
    utils.print_report(train_y, voting_predict(train_x))

    print('\nResults on test:')
    utils.print_report(test_y, voting_predict(test_x))

#lr = models[1]
svc = models[4]
from sklearn.metrics import precision_recall_curve
#precision, recall, thresholds = precision_recall_curve(test_y, lr.predict_proba(train_x)[:,1])
precision, recall, thresholds = precision_recall_curve(test_y, svc.decision_function(test_x))
print(precision)
print(recall)
print(thresholds)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(test_y, svc.decision_function(test_x))
#average_precision = average_precision_score(test_y, lr.predict_proba(train_x)[:,1])

import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVC Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.savefig('SVC_test_PRC.png')
