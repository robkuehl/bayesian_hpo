#%%
# IMPORTS
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join as pathjoin
from sklearn.preprocessing import StandardScaler

# custom modules
from src.data.make_dataset import main_make_dataset

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#%%
main_make_dataset(raw_data_path=Path('../data/raw/OnlineNewsPopularity.csv'), 
                  output_filepath=Path('../data/processed/data_processed.csv'))
#%%
data_preprocessed_df = pd.read_csv('../data/processed/data_processed.csv')

#%%
training_data_df = data_preprocessed_df.drop(columns=[' shares'])
target_var_df = data_preprocessed_df[' shares']
# %%
target_var_df.head()
# %%
