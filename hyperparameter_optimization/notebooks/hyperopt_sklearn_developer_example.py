#%%
import os
os.environ['OMP_NUM_THREADS'] = '1'

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np

#%%
# Download the data and split into training and test sets
iris = load_iris()

X = iris.data
y = iris.target

#%%
test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

#%%
# Instantiate a HyperoptEstimator with the search space and number of evaluations
estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                          preprocessing=any_preprocessing('my_pre'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

#%%
# Search the hyperparameter space based on the data
estim.fit(X_train, y_train)

#%%
# Show the results
print(estim.score(X_test, y_test))

#%%
print(estim.best_model())