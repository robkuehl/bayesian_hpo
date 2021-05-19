#%%
# IMPORTS
import sys
sys.path.append("..")

# custom modules
from src.hpo.hyperopt_sklearn_optimizer import get_regr_estimator
from src.data.load_data import get_processed_data

from hpsklearn import HyperoptEstimator, any_regressor
from hyperopt import tpe
import numpy as np


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# choose kind of estimator
which_estimators = 'custom'
# %%
if which_estimators == 'custom':
    estimator = get_regr_estimator(['random_forest_regression', 'xgboost_regression'], 'myRegr')
elif which_estimators == 'any':
    estimator = HyperoptEstimator(classifier=any_regressor('my_regr'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)
else:
    raise ValueError('Please define which_estimator')
    
#%%
X, y = get_processed_data()
# %%
estimator.fit(X, y)

