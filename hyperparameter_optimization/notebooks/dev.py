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
from src.hpo.hpo_main import hpo_gaussian_process, hpo_hyperopt, create_experiment, run_hpo_search
from hyperopt.pyll import scope
from hyperopt import hp


import mlflow

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



#%%
# process raw data
main_make_dataset(input_filepath=Path('../data/raw/OnlineNewsPopularity.csv'), 
                  output_filepath=Path('../data/processed/data_processed.csv'),
                  overwrite=False)

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
# Create Searchspaces
rf_default_space = get_searchspace(model='random_forest', name='myRegr', task='regr')
rf_uniform_params = dict(
                            n_estimators=scope.int(hp.uniform('myRegr_lightgbm_n_estimators', 25, 2000)),
                            max_features=hp.pchoice('myRegr_random_forest_max_features', [
                                                    (0.2, 'sqrt'),  # most common choice.
                                                    (0.1, 'log2'),  # less common choice.
                                                    (0.1, None),  # all features, less common choice.
                                                    (0.6, hp.uniform('myRegr_random_forest_max_features' + '.frac', 0., 1.))
                                                ]),
                            max_depth=hp.pchoice('myRegr_random_forest_max_depth', [
                                                    (0.7, None),  # most common choice.
                                                    # Try some shallow trees.
                                                    (0.1, 2),
                                                    (0.1, 3),
                                                    (0.1, 4),
                                                ]),
                            min_samples_leaf=1,
                            )
rf_uniform_space = get_searchspace(model='random_forest', name='myRegr', task='regr', **rf_uniform_params)
lgbm_default_space = get_searchspace(model='lightgbm', name='myRegr', task='regr')
lgbm_uniform_params = dict(
                            max_depth=scope.int(hp.uniform('myRegr_lightgbm_max_depth', 1, 11)),
                            num_leaves=scope.int(hp.uniform('myRegr_lightgbm_num_leaves', 2, 121)),
                            learning_rate=hp.uniform('myRegr_lightgbm_learning_rate', 0.0001, 0.5),
                            n_estimators=scope.int(hp.uniform('myRegr_lightgbm_n_estimators', 25, 2000)),
                            min_child_weight=scope.int(hp.uniform('myRegr_lightgbm_min_child_weight', 1, 100)),
                            max_delta_step=0,
                            subsample=hp.uniform('myRegr_lightgbm_subsample', 0.5, 1),
                            colsample_bytree=hp.uniform('myRegr_lightgbm_colsample_bytree', 0.5, 1),
                            reg_alpha=hp.uniform('myRegr_lightgbm_reg_alpha', 0.0001, 1),
                            reg_lambda=hp.uniform('myRegr_lightgbm_reg_lambda', 1, 4),
                            boosting_type=hp.choice('myRegr_lightgbm_boosting_type', ['gbdt', 'dart', 'goss']),
                        )
lgbm_uniform_space = get_searchspace(model='lightgbm', name='myRegr', task='regr', **lgbm_uniform_params)
xgboost_default_space = get_searchspace(model='xgboost', name='myRegr', task='regr')
xgboost_uniform_params = dict(
                            max_depth=scope.int(hp.uniform('myRegr_xgboost_max_depth', 1, 11)),
                            learning_rate=hp.uniform('myRegr_xgboost_learning_rate', 0.0001, 0.5),
                            n_estimators=scope.int(hp.uniform('myRegr_xgboost_n_estimators', 25, 2000)),
                            gamma=hp.uniform('myRegr_xgboost_gamma', 0.0001, 5),
                            min_child_weight=scope.int(hp.uniform('myRegr_xgboost_min_child_weight', 1, 100)),
                            max_delta_step=0,
                            subsample=hp.uniform('myRegr_xgboost_subsample', 0.5, 1),
                            colsample_bytree=hp.uniform('myRegr_xgboost_colsample_bytree', 0.5, 1),
                            colsample_bylevel=hp.uniform('myRegr_xgboost_colsample_bylevel', 0.5, 1),
                            reg_alpha=hp.uniform('myRegr_xgboost_reg_alpha', 0.0001, 1),
                            reg_lambda=hp.uniform('myRegr_xgboost_reg_lambda', 1, 4),
                            scale_pos_weight=1,
                            base_score=0.5,
                            random_state=None
                            )
xgboost_uniform_space = get_searchspace(model='xgboost', name='myRegr', task='regr', **xgboost_uniform_params)

#%%
#define experiments
experiment_dict = [{'model_type':'random_forest', 'space':rf_default_space, 'exp_name':'rf_hpo_default'},
          {'model_type':'random_forest', 'space':rf_uniform_space, 'exp_name':'rf_hpo_uniform'}, 
          {'model_type':'lightgbm', 'space':lgbm_default_space, 'exp_name':'lgbm_hpo_default'},
          {'model_type':'lightgbm', 'space':lgbm_uniform_space, 'exp_name':'lgbm_hpo_uniform'},
          {'model_type':'xgboost', 'space':xgboost_default_space, 'exp_name':'xgboost_hpo_default'},
          {'model_type':'xgboost', 'space':xgboost_uniform_params, 'exp_name':'xgboost_hpo_uniform'},]

#%%
# create experiments for run_hpo
experiments = []
for exp in experiment_dict:
    # tpe experiment
    e = create_experiment(
                algo='tpe',
                task='regr', 
                model_type=exp['model_type'],
                eval_metric='mse',
                num_fold_splits=3,
                param_space=exp['space'],
                X=X,
                y=y,
                max_evals=50,
                experiment_name=exp['exp_name']+'_tpe')
    experiments.append(e)
    # random search experiment
    e = create_experiment(
                algo='random',
                task='regr', 
                model_type=exp['model_type'],
                eval_metric='mse',
                num_fold_splits=3,
                param_space=exp['space'],
                X=X,
                y=y,
                max_evals=50,
                experiment_name=exp['exp_name']+'_random')
    experiments.append(e)


# %%
run_hpo_search(experiments, foldername='hpo_experiment')