#!/usr/bin/python3
import utils

print('Loading dataset...')
(train_x, train_y, test_x, test_y) = utils.prepare_regression_data(True)


models = ['linear_regression.joblib', 'random_forest_shallow_regressor_10.joblib',
'mlp_regressor.joblib', 'random_forest_shallow_regressor_4.joblib',
'mlp_regressor_20_10_5.joblib', 'random_forest_shallow_regressor_8.joblib',
'mlp_regressor_20_10_5_3.joblib', 'ridge.joblib']


from joblib import load
for model in models:
    m = load('annual_regression_models/' + model)
    
    import numpy as np
    print('Train:', model, np.mean(np.square(m.predict(train_x) - train_y)))
    print('Test:', model, np.mean(np.square(m.predict(test_x) - test_y)))

