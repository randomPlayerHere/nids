# %% [markdown]
# # Preprocessing the CICIDS2017 Dataset
# Using OOP Design with Reusable Preprocessor

# %%
import sys
sys.path.append('..')

from preprocessing import CICIDSPreprocessor
import pandas as pd
import numpy as np
import glob
import os

# %% [markdown]
# ## Step 1: Load Raw Data

# %%
# Load all CSV files from raw data directory
day_files = glob.glob("../data/raw/*.csv")
print(f"Found {len(day_files)} CSV files")

df = pd.concat((pd.read_csv(f) for f in day_files), ignore_index=True)
print(f"Total data shape: {df.shape}")
df.head()

# %% [markdown]
# ## Step 2: Initialize and Fit Preprocessor

# %%
# Create preprocessor instance
preprocessor = CICIDSPreprocessor(
    columns_to_drop=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'],
    clip_percentile=0.99,
    random_state=42
)

# Fit and transform the training data with class balancing
X_dcnn, y_labels = preprocessor.fit_transform(
    df,
    with_labels=True,
    balance_classes=True,
    reshape_for_conv1d=True
)

print(f"\nFinal DCNN Input Shape: {X_dcnn.shape}")
print(f"Labels Shape: {y_labels.shape}")

# %% [markdown]
# ## Step 3: Save Processed Data and Preprocessor

# %%
# Create output directory
os.makedirs('../data/processed', exist_ok=True)

# Save processed arrays
np.save('../data/processed/X_dcnn.npy', X_dcnn)
np.save('../data/processed/y_labels.npy', y_labels)
print("✓ Processed arrays saved successfully")

# Save the fitted preprocessor for inference
preprocessor.save('../data/processed/preprocessor.pkl')

# %% [markdown]
# ## Step 4: Test Inference Pipeline

# %%
# Simulate inference on a small sample
from preprocessing import preprocess_for_inference

# Take a small sample for testing
test_sample = df.head(10).copy()

# Preprocess for inference (this is how you'd use it in production)
X_inference = preprocess_for_inference(
    test_sample,
    preprocessor_path='../data/processed/preprocessor.pkl'
)

print(f"Inference output shape: {X_inference.shape}")
print("✓ Inference preprocessing works correctly!")

# %%
# Display preprocessor info
print("\n=== Preprocessor Summary ===")
print(f"Number of features: {preprocessor.num_features}")
print(f"Feature names: {preprocessor.get_feature_names()[:5]}... (showing first 5)")
print(f"Clip percentile: {preprocessor.clip_percentile}")
print(f"Is fitted: {preprocessor.is_fitted}")

# %%



