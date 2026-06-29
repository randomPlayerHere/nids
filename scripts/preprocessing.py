# %%
import pandas as pd
import numpy as np
import re
import glob
import json
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# %% [markdown]
# ### Merging all days into one

# %%
day_files= glob.glob("../data/raw/*.csv")
df = pd.concat((pd.read_csv(f) for f in day_files), ignore_index=True)

print("Dimensions:", df.shape)
df.head()

# %% [markdown]
# ### Dropping Columns

# %%
df.columns = df.columns.str.strip()

to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df = df.drop(columns=to_drop, errors='ignore')

# %% [markdown]
# ### Merging Label Variants

# %%
df['Label'] = df['Label'].str.replace(
    r'Web Attack\s*.\s*Brute Force', 'Web Attacks', regex=True
)
df['Label'] = df['Label'].str.replace(
    r'Web Attack\s*.\s*XSS', 'Web Attacks', regex=True
)
df['Label'] = df['Label'].str.replace(
    r'Web Attack\s*.\s*Sql Injection', 'Web Attacks', regex=True
)

# Bot -> Botnet
df['Label'] = df['Label'].str.replace(r'^Bot$', 'Botnet', regex=True)

print(df['Label'].value_counts())

# %% [markdown]
# ### Drop Extremely Rare Classes

# %%
MIN_SAMPLES = 100

class_counts = df['Label'].value_counts()
rare_classes  = class_counts[class_counts < MIN_SAMPLES].index.tolist()

if rare_classes:
    print(f"\nDropping rare classes (< {MIN_SAMPLES} samples): {rare_classes}")
    df = df[~df['Label'].isin(rare_classes)]
    print("Updated class distribution:")
    print(df['Label'].value_counts())
else:
    print("No rare classes found")

# %% [markdown]
# ### Handling Infinity and NaN

# %%
inf_count = np.isinf(df.select_dtypes(include=[np.number])).values.sum()
print(f"Total count of infinite values: {inf_count}")

print("Replacing all Inf's with NaN")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

before = len(df)
df.dropna(inplace=True)
after = len(df)
print(f"Dropped {before - after:,} rows with NaN")

# %% [markdown]
# ### Clipping Extreme Outliers

# %%
features = df.columns.difference(['Label'])

for col in features:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.clip(df[col], a_min=None, a_max=upper_limit)

print("Outliers clipped.")

# %% [markdown]
# ### Scaling

# %%
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

print("Data scaled between 0 and 1.")

# %% [markdown]
# ### Encode Labels (Multi-Class)

# %%
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

num_classes   = len(le.classes_)
class_mapping = {str(i): name for i, name in enumerate(le.classes_)}

print(f"\nNumber of classes: {num_classes}")
print("Encoded class mapping:")
for k, v in class_mapping.items():
    print(f"  {k} -> {v}")

# %% [markdown]

# %%
X = df.drop('Label', axis=1)
y = df['Label']

CAP_PER_CLASS = 50_000

current_counts = y.value_counts()
sampling_strategy = {}

for class_id, count in current_counts.items():
    if count > CAP_PER_CLASS:
        sampling_strategy[class_id] = CAP_PER_CLASS
    # classes already below cap are left untouched

print("Classes being reduced:", {
    class_mapping[str(k)]: v for k, v in sampling_strategy.items()
})

rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
for class_id, count in resampled_counts.items():
    print(f"  [{class_id}] {class_mapping[str(class_id)]:<20} {count:,} samples")

# %% [markdown]
# ### Reshape for Conv1D

# %%
num_features = X_resampled.shape[1]

X_dcnn = X_resampled.values.reshape(-1, num_features, 1)

print(f"\nFinal DCNN Input Shape: {X_dcnn.shape}")
print(f"Number of output classes: {num_classes}")

# %% [markdown]
# ### Save All Processed Files

# %%
os.makedirs('../data/processed', exist_ok=True)
np.save('../data/processed/X_dcnn.npy', X_dcnn)
np.save('../data/processed/y_labels.npy', y_resampled)
joblib.dump(scaler, '../data/processed/cicids_scaler.pkl')
with open('../data/processed/class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)
# %%
