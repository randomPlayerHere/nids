# Unit testing of this is done along with the EDA
import pandas as pd
import numpy as np

def clean_data(df, duplicates=True, inf_remove=True, missing_val=True, non_unique_col=True, attack_groups=True):
    if duplicates:
        duplicate_row_removal(df)
        duplicate_column_removal(df)
    if inf_remove:
        remove_infinite(df)
    if missing_val:
        missing_val(df)
    if non_unique_col:
        same_val(df)
    if attack_groups(df):
        add_attack_groups(df)

def duplicate_row_removal(df : pd.dfFrame):
    df.drop_duplicates(keep='first')

def duplicate_column_removal(df):
    columns = df.columns
    to_drop = set()
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            if df[col1].equals(df[col2]):
                to_drop.add(col2)
    print(f"Duplicate columns are: {to_drop}")
    df.drop(columns=list(to_drop), inplace=True)

def remove_infinite(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Replaced inf with nan")

def missing_val(df):
    df = df.dropna(inplace=True)
    print("All tuples with Nan dropped")

def same_val(df):
    same_val = []
    for col in df.columns:
        if (len(df[col].unique())) ==1:
            same_val.append(col)
            print(f"{col} has the same unique value for all the tuple")
    df.drop(same_val, axis=1, inplace=True)

def add_attack_groups(df):
    group_mapping = {
    'BENIGN': 'Normal Traffic',
    'DoS Hulk': 'DoS',
    'DDoS': 'DDoS',
    'PortScan': 'Port Scanning',
    'DoS GoldenEye': 'DoS',
    'FTP-Patator': 'Brute Force',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bots',
    'Web Attack � Brute Force': 'Web Attacks',
    'Web Attack � XSS': 'Web Attacks',
    'Infiltration': 'Infiltration',
    'Web Attack � Sql Injection': 'Web Attacks',
    'Heartbleed': 'Miscellaneous'
    }
    df['Attack Type'] = df['Label'].map(group_mapping)
    df.drop(df[(df['Attack Type'] == 'Infiltration') | (df['Attack Type'] == 'Miscellaneous')].index, inplace=True)