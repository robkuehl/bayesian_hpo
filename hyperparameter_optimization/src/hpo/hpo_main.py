import sys
sys.path.append('..')
import os
from getpass import getpass
from src.hpo.hpo_objective import get_optimization_func
from skopt import gp_minimize
from hyperopt import hp, fmin, Trials, tpe, space_eval, rand
import mlflow
from pathlib import Path
from os.path import join as pathjoin
mlruns_folderpath = pathjoin(Path(__file__).parent.absolute(), Path('../../data/mlflow'))
from mlflow.tracking import MlflowClient




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


def hpo_hyperopt(experiment_id, algo, task, model_type, eval_metric, num_fold_splits, param_space, X, y, max_evals):
    
    if algo=='tpe':
        search_algo = tpe.suggest
    elif algo=='random':
        search_algo = rand.suggest
    else:
        raise ValueError('Chosen Algorithm not supported. Must be either tpe or random to use hyperopt backend.')
    
    opt_func = get_optimization_func(search_algo=algo,
                                     experiment_id=experiment_id,
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
        algo=search_algo,
    )
    # hp.choice return index of hyperparameter choice -> turn indices into real parameter outcomes
    result = space_eval(space=param_space, hp_assignment=result)
    
    return result

# https://dagshub.com/robkuehl/hpo_pipeline.mlflow

    
    
    
def run_hpo_search(experiments, foldername):
    mlflow.set_tracking_uri(uri='file:/'+str(pathjoin(mlruns_folderpath, foldername, 'mlruns'))) 
    print('Storing mlflow logs in {}'.format(mlflow.tracking.get_tracking_uri()))
    
    if type(experiments[0])!=dict:
        raise ValueError('Experiments need to be dictionaries!')
    # Create an experiment with a name that is unique and case sensitive.
    client = MlflowClient()
    for experiment in experiments:
        experiment_name = experiment.pop('experiment_name')
        experiment_id = client.create_experiment(experiment_name)
        print(experiment_id)
        client.set_experiment_tag(experiment_id, 'desc', '{}_{}'.format(experiment['model_type'], experiment['algo']))
        if experiment['algo'] in ['tpe', 'random']:
            _ = hpo_hyperopt(experiment_id=experiment_id, **experiment)
            
        else:
            print('Currently only hpyeropt tpe and random supported!')
        
    
    
def create_experiment(experiment_name, algo, task, model_type, eval_metric, num_fold_splits, param_space, X, y, max_evals):
    experiment = {
        'experiment_name':experiment_name, 
        'algo':algo, 
        'task':task,
        'model_type':model_type, 
        'eval_metric':eval_metric, 
        'num_fold_splits':num_fold_splits, 
        'param_space':param_space, 
        'X':X, 
        'y':y, 
        'max_evals':max_evals
    }
    return experiment