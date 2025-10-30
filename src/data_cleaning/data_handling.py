import pandas as pd
import numpy as np
import os

default_path = '../data/raw'

def get_data(directory=default_path):
    data_dict = data_loader(directory)
    return data_concatenate(data_dict)


def data_loader(directory :str) -> dict[str, pd.DataFrame]:
    data_dictionary = {}
    for filename in os.listdir(directory):
        if filename[-4:] == ".csv":
            path= os.path.join(directory, filename)
            df = pd.read_csv(path)
            key = os.path.splitext(os.path.splitext(filename)[0])[0] #this works let it be
            data_dictionary[key] = df
    return data_dictionary

def data_concatenate(data_dict : dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat([df.assign(source=name) for name, df in data_dict.items()],ignore_index=True)
    data.columns = data.columns.str.strip()
    return data

def get_sample(directory=default_path, nrows=1000, fraction=None):
    # 2 choices: nrows and fraaction
    # nrows: choose n rows subset
    #fraction: choose a fraction of rows from each dataset(csv file)
    data_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)
            if fraction:
                df = pd.read_csv(path)
                df = df.sample(frac=fraction, random_state=42)
            else:
                df = pd.read_csv(path, nrows=nrows)
            key = os.path.splitext(os.path.splitext(filename)[0])[0]
            data_dict[key] = df

    return data_concatenate(data_dict)