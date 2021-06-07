from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
from hyperopt.pyll.stochastic import sample


def get_searchspace(model:str, name:str, task:str, **hyperparams) -> dict:
    """function to get the searchspace for hyperopt for a specific model 

    Args:
        model (str): name of the model. Options: xgboost, lightgbm, random_forest
        name (str): name of the specific model e.g. 'myRegr'
        task: (str): regr, binary_clf
        hyperparams (dict) [optional]: fix value or distr. for specific parameters in the searchspace.
                                  Otherweise, default will be chosen.

    Returns:
        hp_space (dict): hyperparameter space for hyperopt hpo
    """
    if task not in ['regr', 'binary_clf']:
            raise ValueError('You need to define the task correct! Either binary_clf or regr.')
        
    def _name_func(msg):
        return '%s.%s_%s' % (name, model, msg)
    
    if model == 'xgboost':
        return _get_xgboost_hp_space(_name_func, task, **hyperparams)
    
    if model == 'random_forest':
        return _get_random_forest_hp_space(_name_func, task, **hyperparams)

    if model == 'light_gbm':
        return _get_lightgbm_hp_space(_name_func, task, **hyperparams)
    
    else:
        print('You did not provide a valid model. Either xgboost, random_forest or light_gbm.')

def get_space_sample(space):
    return sample(space)


def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state
    

# TODO: Anpassen der Default Searchspaces -> 6000 Estimator nicht sinnvoll!
# TODO: Nachschlagen der Verteilungen 
###################################################
##==== XGBoost hyperparameters search space ====##
###################################################

def _xgboost_max_depth(name):
    return scope.int(hp.uniform(name, 1, 11))

def _xgboost_learning_rate(name):
    return hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001

# TODO: Set Max Estimators to a reasonable size
def _xgboost_n_estimators(name):
    #return scope.int(hp.quniform(name, 100, 6000, 200))
    return scope.int(hp.lognormal(name,  mu=0, sigma=0.5)*100)

def _xgboost_gamma(name):
    return hp.loguniform(name, np.log(0.0001), np.log(5)) - 0.0001

def _xgboost_min_child_weight(name):
    return scope.int(hp.loguniform(name, np.log(1), np.log(100)))

def _xgboost_subsample(name):
    return hp.uniform(name, 0.5, 1)

def _xgboost_colsample_bytree(name):
    return hp.uniform(name, 0.5, 1)

def _xgboost_colsample_bylevel(name):
    return hp.uniform(name, 0.5, 1)

def _xgboost_reg_alpha(name):
    return hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001

def _xgboost_reg_lambda(name):
    return hp.loguniform(name, np.log(1), np.log(4))

def _xgboost_hp_space(
    name_func,
    max_depth=None,
    learning_rate=None,
    n_estimators=None,
    gamma=None,
    min_child_weight=None,
    max_delta_step=0,
    subsample=None,
    colsample_bytree=None,
    colsample_bylevel=None,
    reg_alpha=None,
    reg_lambda=None,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=None,
    n_jobs=-1):
    '''Generate XGBoost hyperparameters search space
    '''
    hp_space = dict(
        max_depth=(_xgboost_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        learning_rate=(_xgboost_learning_rate(name_func('learning_rate'))
                       if learning_rate is None else learning_rate),
        n_estimators=(_xgboost_n_estimators(name_func('n_estimators'))
                      if n_estimators is None else n_estimators),
        gamma=(_xgboost_gamma(name_func('gamma'))
               if gamma is None else gamma),
        min_child_weight=(_xgboost_min_child_weight(name_func('min_child_weight'))
                          if min_child_weight is None else min_child_weight),
        max_delta_step=max_delta_step,
        subsample=(_xgboost_subsample(name_func('subsample'))
                   if subsample is None else subsample),
        colsample_bytree=(_xgboost_colsample_bytree(name_func('colsample_bytree'))
                          if colsample_bytree is None else colsample_bytree),
        colsample_bylevel=(_xgboost_colsample_bylevel(name_func('colsample_bylevel'))
                          if colsample_bylevel is None else colsample_bylevel),
        reg_alpha=(_xgboost_reg_alpha(name_func('reg_alpha'))
                   if reg_alpha is None else reg_alpha),
        reg_lambda=(_xgboost_reg_lambda(name_func('reg_lambda'))
                    if reg_lambda is None else reg_lambda),
        scale_pos_weight=scale_pos_weight,
        base_score=base_score,
        seed=_random_state(name_func('rstate'), random_state=random_state),
        n_jobs=n_jobs
    )
    return hp_space

def _get_xgboost_hp_space(_name_func, task, **hyperparams):

    # objective = None
    # if 'objective' in hyperparams.keys():
    #     objective = hyperparams.drop('objective')

    hp_space = _xgboost_hp_space(_name_func, **hyperparams)
    # if task == 'binary_clf':
    #     hp_space['objective'] = ('binary:logistic' if objective==None else objective)
    # elif task == 'regr':
    #     hp_space['objective'] = ('reg:squarederror' if objective==None else objective)
 
    return hp_space
    

####################################################################
##==== Hyperparameter Generators for trees ====##
####################################################################

def _trees_n_estimators(name):
    return scope.int(hp.qlognormal(name, 1, 0.5, 1))
    #return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))

def _trees_clf_criterion(name):
    return hp.choice(name, ['gini', 'entropy'])

def _trees_regr_criterion(name):
    return hp.choice(name, ['mse', 'mae'])

def _trees_max_features(name):
    return hp.pchoice(name, [
        (0.2, 'sqrt'),  # most common choice.
        (0.1, 'log2'),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + '.frac', 0., 1.))
    ])

def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])

def _trees_min_samples_split(name):
    return 2

def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])

def _trees_bootstrap(name):
    return hp.choice(name, [True, False])



####################################################################
##==== Random forest hyperparameters search space ====##
####################################################################
def _trees_hp_space(
        name_func,
        n_estimators=None,
        max_features=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=False
        ):
    '''Generate trees ensemble hyperparameters search space
    '''
    hp_space = dict(
        n_estimators=(_trees_n_estimators(name_func('n_estimators'))
                      if n_estimators is None else n_estimators),
        max_features=(_trees_max_features(name_func('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(name_func('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(name_func('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(name_func('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(name_func('rstate'), random_state),
        verbose=verbose,
    )
    return hp_space



def _get_random_forest_hp_space(_name_func, task, **hyperparams):
    criterion = None
    if 'criterion' in list(hyperparams.keys()):
        criterion = hyperparams.pop('criterion')
    hp_space = _trees_hp_space(_name_func, **hyperparams)
    if task == 'regr':
        hp_space['criterion'] = (_trees_regr_criterion(_name_func('criterion')) if criterion is None else criterion)
    else:
        hp_space['criterion'] = (_trees_clf_criterion(_name_func('criterion')) if criterion is None else criterion)
    return hp_space


    
    
###################################################
##==== LightGBM hyperparameters search space ====##
###################################################

def _lightgbm_max_depth(name):
    return scope.int(hp.uniform(name, 1, 11))

def _lightgbm_num_leaves(name):
    return scope.int(hp.uniform(name, 2, 121))

def _lightgbm_learning_rate(name):
    return hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001

def _lightgbm_n_estimators(name):
    return scope.int(hp.quniform(name, 100, 6000, 200))

def _lightgbm_gamma(name):
    return hp.loguniform(name, np.log(0.0001), np.log(5)) - 0.0001

def _lightgbm_min_child_weight(name):
    return scope.int(hp.loguniform(name, np.log(1), np.log(100)))

def _lightgbm_subsample(name):
    return hp.uniform(name, 0.5, 1)

def _lightgbm_colsample_bytree(name):
    return hp.uniform(name, 0.5, 1)

def _lightgbm_colsample_bylevel(name):
    return hp.uniform(name, 0.5, 1)

def _lightgbm_reg_alpha(name):
    return hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001

def _lightgbm_reg_lambda(name):
    return hp.loguniform(name, np.log(1), np.log(4))

def _lightgbm_boosting_type(name):
    return hp.choice(name, ['gbdt', 'dart', 'goss'])

def _lightgbm_hp_space(
    name_func,
    max_depth=None,
    num_leaves=None,
    learning_rate=None,
    n_estimators=None,
    min_child_weight=None,
    max_delta_step=0,
    subsample=None,
    colsample_bytree=None,
    reg_alpha=None,
    reg_lambda=None,
    boosting_type=None,
    scale_pos_weight=1,
    random_state=None):
    '''Generate LightGBM hyperparameters search space
    '''
    hp_space = dict(
        max_depth=(_lightgbm_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        num_leaves=(_lightgbm_num_leaves(name_func('num_leaves'))
                    if num_leaves is None else num_leaves),
        learning_rate=(_lightgbm_learning_rate(name_func('learning_rate'))
                       if learning_rate is None else learning_rate),
        n_estimators=(_lightgbm_n_estimators(name_func('n_estimators'))
                      if n_estimators is None else n_estimators),
        min_child_weight=(_lightgbm_min_child_weight(name_func('min_child_weight'))
                          if min_child_weight is None else min_child_weight),
        max_delta_step=max_delta_step,
        subsample=(_lightgbm_subsample(name_func('subsample'))
                   if subsample is None else subsample),
        colsample_bytree=(_lightgbm_colsample_bytree(name_func('colsample_bytree'))
                          if colsample_bytree is None else colsample_bytree),
        reg_alpha=(_lightgbm_reg_alpha(name_func('reg_alpha'))
                   if reg_alpha is None else reg_alpha),
        reg_lambda=(_lightgbm_reg_lambda(name_func('reg_lambda'))
                    if reg_lambda is None else reg_lambda),
        boosting_type=(_lightgbm_boosting_type(name_func('boosting_type'))
                    if boosting_type is None else boosting_type),
        scale_pos_weight=scale_pos_weight,
        seed=_random_state(name_func('rstate'), random_state)
    )
    return hp_space


def _get_lightgbm_hp_space(_name_func, task, **hyperparams):
    hp_space = _lightgbm_hp_space(_name_func, **hyperparams)
    # if task == 'regr':
    #     hp_space['objective'] = 'regression'
    # elif task == 'binary_clf':
    #     hp_space['criterion'] = 'binary'
    return hp_space