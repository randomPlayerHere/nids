import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")
st.markdown("### KNN-based Attack Detection using KDD Cup Dataset")
st.markdown("---")

@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load("models/knn_pipeline.pkl")
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

@st.cache_data
def load_test_data():
    columns = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
        'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'
    ]
    try:
        test_df = pd.read_csv("data/KDDTest+.csv", header=None)
        test_df.columns = columns
        test_df['outcome'] = test_df['outcome'].map(lambda a: 'normal' if a == 'normal' else 'attack')
        return test_df
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None

def preprocess_data(df):
    df_copy = df.copy()
    original_label = df_copy['outcome'].copy()
    X = df_copy.drop(['outcome', 'level'], axis=1, errors='ignore')
    return X, original_label

pipeline = load_models()
test_df = load_test_data()

if pipeline is not None and test_df is not None:
    st.sidebar.header("⚙️ Controls")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Single Prediction", "Batch Prediction", "Random Sample"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Total test samples: {len(test_df)}")
    if mode == "Single Prediction":
        st.subheader("🔍 Single Connection Analysis")
        row_index = st.number_input(
            "Enter row index to predict:",
            min_value=0,
            max_value=len(test_df)-1,
            value=0,
            step=1
        )
        if st.button("🚀 Predict", type="primary"):
            sample = test_df.iloc[[row_index]]
            actual_label = sample['outcome'].values[0]
            X, _ = preprocess_data(sample)
            prediction = pipeline.predict(X)[0]
            predicted_label = 'normal' if int(prediction) == 0 else 'attack'
            proba = None
            if hasattr(pipeline, 'predict_proba'):
                try:
                    proba = pipeline.predict_proba(X)[0]
                except Exception:
                    proba = None
            confidence = max(proba) * 100 if proba is not None else None
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Label", actual_label)
            with col2:
                st.metric("Predicted Label", predicted_label)
            with col3:
                st.metric("Confidence", f"{confidence:.2f}%" if confidence is not None else "N/A")
            if actual_label == predicted_label:
                st.success("✅ Correct Prediction!")
            else:
                st.error("❌ Incorrect Prediction")
            st.markdown("### 📋 Connection Details")
            st.dataframe(sample, use_container_width=True)
    elif mode == "Batch Prediction":
        st.subheader("📊 Batch Connection Analysis")
        num_samples = st.slider(
            "Number of samples to predict:",
            min_value=10,
            max_value=min(1000, len(test_df)),
            value=100,
            step=10
        )
        start_index = st.number_input(
            "Starting index:",
            min_value=0,
            max_value=len(test_df)-num_samples,
            value=0,
            step=1
        )
        if st.button("🚀 Predict Batch", type="primary"):
            batch = test_df.iloc[start_index:start_index+num_samples]
            X, original_labels = preprocess_data(batch)
            predictions = pipeline.predict(X)
            predicted_labels = ['normal' if int(p) == 0 else 'attack' for p in predictions]
            actual_binary = original_labels.values
            correct = sum(actual_binary == predicted_labels)
            accuracy = (correct / len(predicted_labels)) * 100
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", num_samples)
            with col2:
                st.metric("Correct Predictions", correct)
            with col3:
                st.metric("Incorrect Predictions", num_samples - correct)
            with col4:
                st.metric("Accuracy", f"{accuracy:.2f}%")
            results_df = batch.copy()
            results_df['Predicted'] = predicted_labels
            results_df['Correct'] = (actual_binary == results_df['Predicted'])
            st.markdown("### 📋 Prediction Results (First 20)")
            display_df = results_df[['duration', 'protocol_type', 'service', 'flag', 
                                     'src_bytes', 'dst_bytes', 'outcome', 'Predicted', 'Correct']].head(20)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("### 📈 Prediction Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Actual Labels:**")
                st.write(pd.Series(actual_binary).value_counts())
            with col2:
                st.write("**Predicted Labels:**")
                st.write(pd.Series(predicted_labels).value_counts())
    else:
        st.subheader("🎲 Random Sample Prediction")
        num_random = st.slider(
            "Number of random samples:",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        if st.button("🎲 Generate & Predict", type="primary"):
            random_indices = np.random.choice(len(test_df), num_random, replace=False)
            random_samples = test_df.iloc[random_indices]
            X, original_labels = preprocess_data(random_samples)
            predictions = pipeline.predict(X)
            predicted_labels = ['normal' if int(p) == 0 else 'attack' for p in predictions]
            actual_binary = original_labels.values
            results_df = random_samples.copy()
            results_df['Predicted'] = predicted_labels
            results_df['Correct'] = (actual_binary == results_df['Predicted'])
            correct = sum(results_df['Correct'])
            accuracy = (correct / num_random) * 100
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correct", f"{correct}/{num_random}")
            with col2:
                st.metric("Accuracy", f"{accuracy:.2f}%")
            st.markdown("### 📋 Results")
            display_df = results_df[['duration', 'protocol_type', 'service', 'src_bytes', 
                                     'dst_bytes', 'outcome', 'Predicted', 'Correct']]
            st.dataframe(display_df, use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.info(f"**Algorithm:** K-Nearest Neighbors\n\n**Features:** PCA (20 components)\n\n**Classes:** Normal (0), Attack (1)")
else:
    st.error("⚠️ Failed to load models or test data. Please ensure the models are trained and data files exist.")
    st.info("Run the training script first to generate model files in the `models/` directory.")