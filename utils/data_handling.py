import pandas as pd
import os

def get_data(directory : str):
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