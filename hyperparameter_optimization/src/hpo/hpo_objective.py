import sys
sys.path.append('..')

from functools import partial


from src.models.model_evaluation import kfold_validation

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import mlflow
from datetime import datetime
import json


def get_model(model_type:str, task, hyperparams):
    if task == 'regr':
        if model_type == 'random_forest':
            return RandomForestRegressor(**hyperparams)
        elif model_type == 'xgboost':
            return XGBRegressor(**hyperparams)
        elif model_type == 'lightgbm':
            return LGBMRegressor(**hyperparams)
    elif task == 'binary_clf':
        if model_type == 'random_forest':
            return RandomForestClassifier(**hyperparams)
        elif model_type == 'xgboost':
            return XGBClassifier(**hyperparams)
        elif model_type == 'lightgbm':
            return LGBMClassifier(**hyperparams)


def skopt_objective(task:str, model_type:str, eval_metric:str, X, y, param_names, params):
    """Objective function for skopt hyperparameter optimization

    Args:
        task (str): 'regr' or 'binary_clf'
        model (sklearn model object): model which should be optimized regarding its hyperparams.
                                    Options: xgboost, random_forest, light_gbm
        eval_metric (str): metric for the evaluation. 
                        - regression: mse, mae, mape.
        X (numpy.array): Input data for the model
        y (numpy.array): target values to predict
        param_names (list): names of the hyperparams
        params (list): values of the hyperparams respectively

    Returns:
        fold_score (float): avergae of metric for folds of crossvalidation
    """
    hyperparams = dict(zip(param_names, params))
    model = get_model(model_type, task, hyperparams)
    fold_score = kfold_validation(task, model, X, y, eval_metric)
    
    if task == 'regr':
        # fold_score = mean of errors -> Minimize
        return fold_score
    
    
def hyperopt_objective(task:str, model_type, eval_metric:str, X, y, hyperparams:dict):
    '''
    fold_score = 0
    with mlflow.start_run():
        model = get_model(model_type, task, hyperparams)
        fold_score = kfold_validation(task, model, X, y, eval_metric)
        mlflow.log_param('model_type', model_type)
        mlflow.log_metric('fold_score', fold_score)
        mlflow.log_params(hyperparams)
    '''
    print(hyperparams)
        
    model = get_model(model_type, task, hyperparams)
    fold_score = kfold_validation(task, model, X, y, eval_metric)
    
    # TODO: Loggen der Hyperparameter pro run mit Score
    
    if task == 'regr':
        # fold_score = mean of errors -> Minimize
        return fold_score
    
    
    
    
def get_optimization_func(search_algo, task, model_type, eval_metric, X, y, param_names=None):
    if search_algo == 'gaussian_processes':
        if param_names==None:
            raise ValueError('Need to provide param_names!')
        opt_func = partial(skopt_objective, task, model_type, eval_metric, X, y, param_names)
        return opt_func
    
    if search_algo == 'tpe':
        opt_func = partial(hyperopt_objective, task, model_type, eval_metric, X, y)
        return opt_func
    
    
    

        