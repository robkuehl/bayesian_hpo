# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
#from dotenv import find_dotenv, load_dotenv
from src.functions.helper_functions import export_features, load_var_types, dfinfo
from sklearn.preprocessing import StandardScaler


#@click.command()
#@click.argument('raw_data_path', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main_make_dataset(raw_data_path, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # load raw data
    data_df = pd.read_csv(raw_data_path)
    
    # folderpath for raw data folder
    raw_data_folderpath = Path(raw_data_path).parent.absolute()
    
    # load table with variables and their dtypes
    var_types_df = load_var_types()
    
    # create lists of variables by dtype
    numeric_vars_list = var_types_df[var_types_df['var_type']=='numeric']['Variable'].to_list()
    boolean_vars_list = var_types_df[var_types_df['var_type']=='bool']['Variable'].to_list()
    
    # scale numeric variables
    scaler = StandardScaler()
    numeric_data_df = data_df[numeric_vars_list]
    scaled_numeric_data_df = pd.DataFrame(data=scaler.fit_transform(numeric_data_df), 
                                        index=numeric_data_df.index, 
                                        columns=numeric_data_df.columns)
    
    # concat df of different dtypes
    data_preprocessed_df = pd.concat([scaled_numeric_data_df, data_df[boolean_vars_list]], axis=1)
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
