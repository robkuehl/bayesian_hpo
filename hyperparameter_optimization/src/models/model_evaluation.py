from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import sys
sys.path.append('..')
from src.functions.helper_functions import load_obj
from os.path import join as pathjoin
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np


def evaluate_reg_model(model, metric,  X_test, y_test, rescaled):
    """Function to evaluate model performance of a regressor. Metric can be chosen.

    Args:
        model: Regression model object with predict method (see sklearn for examples)
        metric (str): Metric which should be used for evaluation. Choose between mse, mae, mape
        X_test (np.array): numpy array of input data for testing
        y_test (np.array): numpy value of target variable for prediction
        rescaled (bool): If 'True', data will be rescaled before evaluation

    Raises:
        ValueError: Raised of chosen metric is not a possible choice.

    Returns:
        error [float]: error depedning on the chosen metric
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


def kfold_validation(task, model, X, y, metric, n_splits):
    """Function to run k-fold cross validation on a model. Model for evaluation depends on the task.

    Args:
        task (str): 'regr' or 'binary_clf'
        model: model object with fit function (see sklearn) which only needs X and y as input
        X_train (np.array): numpy array of input data for training
        y_train (np.array): numpy value of target variable for prediction
        metric (str): Metric which should be used for evaluation. Choose between mse, mae, mape
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    
    scores = {}
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
            key = '{}_{}_{}'.format(metric, 'fold', iter)
            scores[key]=error
            
    return scores, np.asarray(scores.values).mean()
        
        