import sys
sys.path.append('..')

from src.hpo.hpo_objective import get_optimization_func
from skopt import gp_minimize
from hyperopt import hp, fmin, Trials, tpe, space_eval
import mlflow


def hpo_gaussian_process(task, model, eval_metric, param_space, X, y, n_calls, n_random_starts, verbose):
    param_names=list(param_space.keys())
    dimensions=list(param_space.values())
    opt_func = get_optimization_func(search_algo='gaussian_processes', 
                                     task=task, 
                                     model=model, 
                                     eval_metric=eval_metric, 
                                     param_names=param_names, 
                                     X=X, 
                                     y=y)
    
    result = gp_minimize(opt_func, 
                         dimensions=dimensions, 
                         n_calls=n_calls, 
                         n_random_starts=n_random_starts, 
                         verbose=verbose)
    
    return dict(zip(param_names, result.x))


def hpo_tpe(task, model_type, eval_metric, num_fold_splits, param_space, X, y, max_evals, experiment_name):
    
    mlflow.set_experiment(experiment_name)
    
    opt_func = get_optimization_func(search_algo='tpe',
                                     task=task, 
                                     model_type=model_type, 
                                     eval_metric=eval_metric, 
                                     X=X, 
                                     y=y,
                                     n_splits=num_fold_splits)
    
    
    trials = Trials()
    
    result = fmin(
        fn=opt_func,
        space=param_space,
        trials=trials,
        max_evals=max_evals,
        algo=tpe.suggest,
    )
    # hp.choice return index of hyperparameter choice -> turn indices into real parameter outcomes
    result = space_eval(space=param_space, hp_assignment=result)
    
    return result
    


    