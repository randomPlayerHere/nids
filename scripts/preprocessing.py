# %% [markdown]
# # Preprocessing the CICIDS2017 Dataset

# %%
import pandas as pd
import numpy as np
import glob

# %% [markdown]
# ### Merging all days into one

# %%
day_files = glob.glob("../data/raw/*.csv")
df = pd.concat((pd.read_csv(f) for f in day_files),ignore_index=True)

# %%
print("Dimesnsions:",df.shape)
df.head()

# %%


# %% [markdown]
# ### Dropping Columns

# %%
df.columns

# %%
df.columns = df.columns.str.strip()

# %%
to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df = df.drop(columns=to_drop,errors='ignore')

# %%
df.columns

# %% [markdown]
# ### Binary Label Mapping

# %%
df['Label'] = (df['Label'] != 'BENIGN').astype(int)

# %%
df['Label'].value_counts()

# %% [markdown]
# ### Handling Infinity and NaN

# %%
inf_count = np.isinf(df).values.sum()
print(f"Total count of infinite values:{inf_count}")

# %%
print("Replacing all Inf's with NaN")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%
before = len(df)
df.dropna(inplace=True)
after = len(df)
print(f"Dropped {before - after:,} rows")

# %% [markdown]
# ### Clipping Extreme Outliers

# %%
features = df.columns.difference(['Label'])

for col in features:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.clip(df[col], a_min=None, a_max=upper_limit)
    
print("Outliers clipped.")

# %%
df.columns.value_counts()

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

print("Data scaled between 0 and 1.")

# %% [markdown]
# ### Handling class Imbalance

# %%
from imblearn.under_sampling import RandomUnderSampler

X = df.drop('Label', axis=1)
y = df['Label']

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("New class distribution:")
print(y_resampled.value_counts())

# %%
num_features = X_resampled.shape[1]

# Reshape for Conv1D: (Samples, Features, 1)
X_dcnn = X_resampled.values.reshape(-1, num_features, 1)

print(f"Final DCNN Input Shape: {X_dcnn.shape}")

# %%
import numpy as np
import joblib
import os

os.makedirs('../data/processed', exist_ok=True)

np.save('../data/processed/X_dcnn.npy', X_dcnn)
np.save('../data/processed/y_labels.npy', y_resampled)
print("Processed arrays saved successfully.")

joblib.dump(scaler, '../data/processed/cicids_scaler.pkl')
print("Scaler saved successfully.")

# %%



