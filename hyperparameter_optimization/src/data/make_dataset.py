# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import os
import pickle
import sys
#from dotenv import find_dotenv, load_dotenv
sys.path.append('..')
from src.functions.helper_functions import export_features, load_var_types, store_obj
from sklearn.preprocessing import StandardScaler
from os.path import join as pathjoin


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main_make_dataset(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if os.path.isfile(output_filepath):
        overwrite = input("Outputfile already exist. If you want to overwrite please type \"yes\"")
        if overwrite!='yes':
            return
        
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # load raw data
    data_df = pd.read_csv(input_filepath)
    
    # load table with variables and their dtypes
    var_types_df = load_var_types()
    
    # create lists of variables by dtype
    numeric_vars_list = var_types_df[var_types_df['var_type']=='numeric']['Variable'].to_list()
    boolean_vars_list = var_types_df[var_types_df['var_type']=='bool']['Variable'].to_list()
    
    # split numeric data into target var and input x
    x_numeric_df = data_df[numeric_vars_list].drop(columns=[' shares'])
    y_df = data_df[[' shares']]
    
    std_scaler = StandardScaler()
    # scale numeric imput variables
    std_scaler_x = std_scaler.fit(x_numeric_df)
    x_numeric_scaled = std_scaler_x.transform(x_numeric_df)
    x_numeric_scaled_df = pd.DataFrame(data=x_numeric_scaled, 
                                        index=x_numeric_df.index, 
                                        columns=x_numeric_df.columns)
    # scale target variable
    std_scaler_y = std_scaler.fit(y_df)
    y_scaled = std_scaler_y.transform(y_df)
    y_scaled_df = pd.DataFrame(data=y_scaled, 
                               index=y_df.index, 
                               columns=y_df.columns)
    
    # save scaler objects
    path = pathjoin(Path(__file__).parent.absolute(), 
                           Path('../../serialized_objects/std_scaler_x.pkl'))
    store_obj(std_scaler_x, path)
    
    path = pathjoin(Path(__file__).parent.absolute(), 
                           Path('../../serialized_objects/std_scaler_y.pkl'))
    store_obj(std_scaler_y, path)
    
    # concat df of different dtypes
    data_preprocessed_df = pd.concat([x_numeric_scaled_df, data_df[boolean_vars_list], y_scaled_df], axis=1)
    # safe preprocessed data
    data_preprocessed_df.to_csv(output_filepath)

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_make_dataset()
