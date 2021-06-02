from hpsklearn import HyperoptEstimator
from hyperopt import hp, tpe
from hpsklearn.components import *
import xgboost



#########################################Searchspaces##########################################
def get_model(model:str, name:str, kwargs:dict):
    name = name+'.{}'.format(model)
    if model == 'knn_regression':
        return knn_regression(name, **kwargs)
    elif model == 'ada_boost_regression':
        return ada_boost_regression(name, **kwargs)
    elif model == 'gradient_boosting_regression':
        return gradient_boosting_regression(name, **kwargs)
    elif model == 'random_forest_regression':
        return random_forest_regression(name, **kwargs)
    elif model == 'extra_trees_regression':
        return extra_trees_regression(name, **kwargs)
    elif model == 'sgd_regression':
        return sgd_regression(name, **kwargs)
    elif model == 'xgboost_regression':
        return xgboost_regression(name, **kwargs)
    else:
        raise ValueError()
    
    
    
def get_regr_estimator(models:list, 
                       name:str, 
                       custom_hyperparams:dict={}, 
                       probs:dict=None, 
                       max_eval:int=120, 
                       trial_timeout:int=120,
                       algo=tpe.suggest) -> HyperoptEstimator:
    """This function returns an Hyperopt-Sklearn Optimizer for a list of models

    Args:
        model (str): svr, svr_linear, svr_rbf, svr_poly, svr_sigmoid, knn_regression, ada_boost_regression,
                     gradient_boosting_regression, random_forest_regression, extra_trees_regression, c,
                     xgboost_regression
        name (str): model name
        custom_hyperparams (dict, optional): dict of dicts. Should contain a dict of hyperparams for every model.
        probs (dict, optional): Key: model, Value: Probability that model is chosen

    Returns:
        estimator (HyperoptEstimator)
    """
    regr_list = []
    
    for model in models:
        if model in list(custom_hyperparams.keys()):
            try:
                regr = get_model(model=model, name=name, kwargs=custom_hyperparams[model])
            except ValueError:
                print('Your chosen model {} is not yet supported and will not be part of model selection!'.format(model))
        else:
            try:
                regr = get_model(model=model, name=name, kwargs={})
            except ValueError:
                print('Your chosen model {} is not yet supported and will not be part of model selection!'.format(model))
        if probs != None:
            regr_list.append((probs[model], regr))
        else: 
            regr_list.append(regr)
            
    if probs == None:
        regressors = hp.choice('%s' % name, regr_list)
    else:
        regressors = hp.pchoice('%s' % name, regr_list)
        
    print(regressors)
        
    estimator = HyperoptEstimator(regressor=regressors)
    return estimator
        
