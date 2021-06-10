from pathlib import Path
import os
from os.path import join as pathjoin
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from shutil import make_archive
import boto3

raw_data_folderpath = pathjoin(Path(__file__).parent.absolute(), Path('../../data/raw'))


#==================================Preprocessing=========================================================
def export_features(df, overwrite=False):
    """Function to export all column indexes from a dataframe to csv
    """
    filepath = pathjoin(raw_data_folderpath, 'var_types.csv')
    if not os.path.isfile(filepath) or overwrite==True:
        pd.DataFrame(index=df.columns).to_csv(filepath)

def load_var_types():
    filepath = pathjoin(raw_data_folderpath, 'var_types.csv')
    if os.path.isfile(filepath):
        var_types = pd.read_csv(filepath, delimiter=';')
        return var_types
    else:
        raise FileNotFoundError("Nessecary file does not exist. Please check of var_types.csv in raw folder!")
    

def store_obj(obj, path):
    print('Storing {}'.format(obj))
    with open(path, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)

        
def load_obj(path):
    with open(path, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
    return obj
    
    
#==================================Training=========================================================
def get_training_data(training_data_df):
    X_df = training_data_df.drop(columns=[' shares'])
    y_df = training_data_df[' shares']
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
    
    
#==================================EDA=========================================================
def dfinfo(df, feature_dtype_list=[], jupyter=False, save=False, returndf=False, filename="None"):
    """ Prints tabular infos to the Dataframe columns
        Parameters:
            df (Pandas Dataframe):  Dataframe to process
            jupyter (Boolean): Set True to display inside notebook, else to print in console
            save (Boolean): If True, the result gets saved to '../reports/csv/<filename>_metadata.csv'
            returndf (Boolean): Switch for returning dataframe
            filename (String): name for saving file
        Returns: Dataframe
    """
    pd.options.display.float_format = '{:.2f}'.format
    table = []
    numrows = df.shape[0]
    toFilter=feature_dtype_list
    print("Number of rows: ", numrows)
    print("-----------------------------")
    for col in df.columns.values:
        df2 = df[[col]]
        coldict = {}
        coldict["Feature"] = col
        coldict["Missing"] = df[col].isnull().sum()
        coldict["Missing %"] = str(round(df[col].isnull().sum() / numrows * 100, 2))+"%"
        coldict["Zeros %"] = str(round(df2[df2[col] == 0].shape[0] / numrows * 100, 2))+"%"
        coldict["min"] = pd.to_numeric(df2[col], errors='coerce').min()
        coldict["max"] = pd.to_numeric(df2[col], errors='coerce').max()
        coldict["Type"] = df[col].dtypes
        # Anpassung an PiSta Start
        if col not in toFilter:
            coldict["nUnique"] = df[col].nunique()
            coldict["Example"] = df[col].unique()[:3]
        else:
            coldict["nUnique"] = None
            coldict["Example"] = None
        # Anpassung an PiSta Ende
        table.append(coldict)
    titles = ['Feature', 'Missing', 'Missing %', 'Zeros %', 'min', 'max', 'Type', 'nUnique', "Example"]
    tableframe = pd.DataFrame(table, columns=titles)
    if jupyter: 
        display(tableframe)
    else: 
        print(tableframe)
    if save: tableframe.to_csv(f"{filename}_metadata.csv", sep=';', encoding='utf-8', index=False,  decimal=',')
    if returndf: 
        for col in tableframe.columns:
            if "%" in col:
                tableframe[col]=tableframe[col].str.replace("%","").astype(float)
        return tableframe
    
    
    #=====================================EXPORT=========================================================
    def zip_folder(root_dir:Path, target_folder:Path, archive_name:str):
        archive_path = pathjoin(target_folder, archive_name)
        make_archive(archive_path, 'zip', root_dir)
        if os.path.isfile(archive_path+'.zip'):
            # successfull compression
            return archive_path+'.zip'
        else:
            # compression failed
            return 1
    
    def upload_to_s3(sourcefile_path:Path, s3_client, s3_bucket, target_path):
        pass
    
    def load_aws_credentials(self, config_file=None, key=None, secret=None):
        """
        Sets the credentials to AWS either by providing the key and secret directly
        or by specifying the location/file containing the confiurations

        config_file : str, optional
            The path to the config file for the AWS credentials, eiher use config_file or key and secret
            config_file overrides key and secret.
        key : str, optional
            The key for the AWS access, either use key and secret or the config_file option
        secret : str, optional
            The secret for the AWS access, either use key and secret or the config_file option
        """
        if config_file is not None:
            key, secret = self._load_aws_config(config_file)

        # XOR testing key and secret
        if bool(key is None) != bool(secret is None):
            self.logger.log_warning("key and secret have to be both either be set or left empty, aborting upload")
            return

        self.key = key
        self.secret = secret