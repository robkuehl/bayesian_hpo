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

# modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#%%
# process raw data
main_make_dataset(input_filepath=Path('../data/raw/OnlineNewsPopularity.csv'), 
                  output_filepath=Path('../data/processed/data_processed.csv'))

#%%
linear_regr = LinearRegression()
random_forest_regr = RandomForestRegressor()
adaboost_regr = AdaBoostRegressor(base_estimator=DecisionTreeClassifier(), n_estimators=200)

model_dict = {
    'linear_regr':linear_regr,
    'random_forest_regr':random_forest_regr,
    'adaboost_regr':adaboost_regr
}

#%%
scores_dict = model_selection.run('regression', model_dict, 'mape')

# %%
for key in scores_dict:
    print('{} has an average error of {}%'.format(key, scores_dict[key]))




# %%
