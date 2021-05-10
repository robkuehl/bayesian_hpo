#%%
# IMPORTS
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from pathlib import Path

# custom modules
from src.data.make_dataset import main_make_dataset
from src.models import model_selection
from src.hpo.skopt_optimizer import get_regr_estimator
from src.data.load_data import get_processed_data

# modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

import hyperopt as hp

from hpsklearn import HyperoptEstimator, any_regressor
from hyperopt import tpe
import numpy as np


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


"""
#%%
# process raw data
main_make_dataset(input_filepath=Path('../data/raw/OnlineNewsPopularity.csv'), 
                  output_filepath=Path('../data/processed/data_processed.csv'))

#%%
linear_regr = LinearRegression()
random_forest_regr = RandomForestRegressor()
adaboost_regr = AdaBoostRegressor(base_estimator=DecisionTreeClassifier(), n_estimators=200)

model_dict = {
    'linear_regr':linear_regr,
    'random_forest_regr':random_forest_regr,
    'adaboost_regr':adaboost_regr
}

#%%
scores_dict = model_selection.run('regression', model_dict, 'mape')

# %%
for key in scores_dict:
    print('{} has an average error of {}%'.format(key, scores_dict[key]))

"""

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

