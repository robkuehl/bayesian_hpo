from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import sys
sys.path.append('..')
from src.functions.helper_functions import load_obj
from os.path import join as pathjoin
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np


def evaluate_reg_model(model, metric,  X_test, y_test, rescaled):
    """Function to evaluate model performance. Metric can be chosen.

    Args:
        model: [description]
        metric (str): Metric which should be used for evaluation. Choose between mse, 
        X_test (np.array): [description]
        y_test (np.array): [description]

    Raises:
        ValueError: Raised of chosen metric is not a possible choice.

    Returns:
        error [float]: error return by the metric
    """
    if rescaled:
        # load scaler
        path = pathjoin(Path(__file__).parent.absolute(), '..', '..', 'serialized_objects', 'std_scaler_y.pkl')
        std_scaler_y = load_obj(path)
        
    # prediction
    y_pred = model.predict(X_test)
    if rescaled:
        # rescale data
        y_pred = std_scaler_y.inverse_transform(y_pred)
        y_test = std_scaler_y.inverse_transform(y_test)
        
    if metric=='mse':
        error = mean_squared_error(y_test, y_pred)
    elif metric=='mae':
        error = mean_absolute_error(y_test, y_pred)
    elif metric=='mape':
        error = mean_absolute_percentage_error(y_test, y_pred)
    elif metric=='rmse':
        pass
    else:
        raise ValueError('You did not give a valid metric for the evaluation!')
    return error


def kfold_validation(task, model, X, y, metric):
    """Function to run k-fold cross validation on a model. Model for evaluation depends on the task.

    Args:
        task (str): 'regression' or 'classification'
        model: model with fit function which only needs X and y
        X_train (np.array): Training Input
        y_train (np.array): Ground Truth
        metric (str): Supported metric (mse, mae, mape)
    """
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    
    scores = []
    iter=0
    for train_index, test_index in kf.split(X):
        iter+=1
        print('Fold Nr.{}'.format(iter))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        
        # evaluation 
        if task=='regr':
            error = evaluate_reg_model(model, metric, X_test, y_test, rescaled=False)
            scores.append(error)
            
    return np.asarray(scores).mean()
        
        