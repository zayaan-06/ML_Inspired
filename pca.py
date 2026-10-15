#!/usr/bin/python3
import utils

import numpy as np
from sklearn.decomposition import PCA

train_x, train_y, test_x, test_y = utils.prepare_data(load_existing=True)
pca = PCA(n_components=0.95, svd_solver='full')
pca.fit(train_x)

print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

print(pca.singular_values_)

print('num_components:', len(pca.explained_variance_ratio_))

