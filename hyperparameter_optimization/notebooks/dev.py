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
from src.data.load_data import get_processed_data

# modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

from src.hpo.hyperopt_searchspaces import get_searchspace, get_space_sample
from src.hpo.hpo_main import hpo_gaussian_process, hpo_tpe

import mlflow

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



#%%

# process raw data
main_make_dataset(input_filepath=Path('../data/raw/OnlineNewsPopularity.csv'), 
                  output_filepath=Path('../data/processed/data_processed.csv'))

#%%
X, y = get_processed_data()

#%%
# linear_regr = LinearRegression()
# random_forest_regr = RandomForestRegressor()
# adaboost_regr = AdaBoostRegressor(base_estimator=DecisionTreeClassifier(), n_estimators=200)

# model_dict = {
#     'linear_regr':linear_regr,
#     'random_forest_regr':random_forest_regr,
#     'adaboost_regr':adaboost_regr
# }

#%%
#scores_dict = model_selection.run('regression', model_dict, 'mape')

# %%
#for key in scores_dict:
#    print('{} has an average error of {}%'.format(key, scores_dict[key]))


#%% 
hyperparams={'verbose':2}
param_space = get_searchspace(model='random_forest', name='myRegr', task='regr', **hyperparams)

#%%

result = hpo_tpe(task='regr', 
                model_type='random_forest',
                eval_metric='mse',
                num_fold_splits=3,
                param_space=param_space,
                X=X,
                y=y,
                max_evals=50,
                experiment_name='random_forest_hpo')

print(result)

# %%
