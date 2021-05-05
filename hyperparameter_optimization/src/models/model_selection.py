import sys
sys.path.append('../..')

from src.models.model_evaluation import kfold_validation
from src.data.load_data import get_processed_data


    
def run(task, model_dict:dict, metric):
    X, y = get_processed_data()
    score_dict = {}
    for key in list(model_dict.keys()):
        print('Evaluate {}'.format(key))
        model = model_dict[key]
        score_dict[key] = kfold_validation(task, model, X, y, metric)
    return score_dict
            