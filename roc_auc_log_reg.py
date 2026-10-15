import utils
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import load

(train_x, train_y, test_x, test_y) = utils.prepare_data(True)
m = load('new_models/log_reg.joblib')

train_proba = m.predict_proba(train_x)[:,1]
test_proba = m.predict_proba(test_x)[:,1]

train_fpr, train_tpr, train_threshold = roc_curve(train_y, train_proba)
test_fpr, test_tpr, test_threshold = roc_curve(test_y, test_proba)

train_roc = roc_auc_score(train_y, train_proba)
test_roc = roc_auc_score(test_y, test_proba)
print('train_roc =', train_roc)
print('test_roc =', test_roc)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(test_fpr, test_tpr, 'b', label = 'AUC = %0.2f' % test_roc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('logreg_roc_test.svg')

print(test_threshold)
