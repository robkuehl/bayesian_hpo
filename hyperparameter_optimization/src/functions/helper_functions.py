from pathlib import Path
import os
from os.path import join as pathjoin
import pandas as pd

raw_data_folderpath = pathjoin(Path(__file__).parent.absolute(), Path('../../data/raw'))

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
        raise FileNotFoundError("Recommended file does not exist. Please check of var_types.csv in raw folder!")
    

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