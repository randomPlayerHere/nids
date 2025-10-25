import pandas as pd
import os

def data_loader(directory :str) -> dict[str, pd.DataFrame]:
    data_dictionary = {}
    for filename in os.listdir(directory):
        if filename.endswith() == ".csv":
            path= os.path.join(directory, filename)
            df = pd.read_csv(path)
            key = os.path.splitext(filename)
            data_dictionary[key] = df
    return data_dictionary

def data_concatenate(data_dict : dict[str, pd.DataFrame]) -> pd.DataFrame:
    