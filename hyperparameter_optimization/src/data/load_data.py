import sys
sys.path.append('..')

from pathlib import Path
from os.path import join as pathjoin

import pandas as pd 

def get_processed_data():
    data_path = pathjoin(Path(__file__).parent.absolute(), Path('../../data/processed/data_processed.csv'))
    processed_data = pd.read_csv(data_path)
    X = processed_data.drop(columns=[' shares'])
    y = processed_data[[' shares']]
    return X, y