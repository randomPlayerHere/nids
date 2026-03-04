"""
Network Intrusion Detection System (NIDS) - Streamlit Interface
Provides an intuitive web interface for network traffic analysis using DCNN
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import traceback

def preprocess_inference(df):
    """Preprocess data for inference"""
    # Drop identifier columns
    to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    df = df.drop(columns=to_drop, errors='ignore')
    
    # Drop Label column if present (not needed for inference)
    df = df.drop(columns=['Label'], errors='ignore')
    
    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    
    return df

# --- Page Configuration ---
st.set_page_config(
    page_title="NIDS Deep Learning Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System (DCNN)")
st.markdown("""
Upload a CSV file containing network traffic flows to detect malicious activity.
The system uses a Deep Convolutional Neural Network trained on the CICIDS2017 dataset.
""")

# --- Constants ---
MODEL_PATH = 'models/nids_dcnn_model.h5'
SCALER_PATH = 'models/cicids_scaler.pkl'
COLUMNS_TO_DROP = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
LABEL_MAP = {0: "✅ BENIGN", 1: "🚨 ATTACK"}

# --- Load Model and Scaler ---
@st.cache_resource
def load_nids_assets():
    """Load model and scaler once and cache them"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# Load assets
model, scaler, error = load_nids_assets()

if error:
    st.error(f"❌ Failed to load ML assets: {error}")
    st.stop()
else:
    st.success("✅ Model and Scaler loaded successfully!")

# --- Sidebar Information ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **NIDS DCNN Model**
    - Architecture: 1D Convolutional Neural Network
    - Dataset: CICIDS2017
    - Classes: Binary (BENIGN vs ATTACK)
    
    **Expected Input:**
    - CSV file with network flow features
    - Same features as training data
    - Columns like Duration, Flow Bytes/s, etc.
    
    **Columns Automatically Dropped:**
    - Flow ID
    - Source IP
    - Destination IP  
    - Timestamp
    - Label (if present)
    """)
    
    st.header("📊 Expected Features")
    st.info(f"The model expects **{scaler.n_features_in_}** features after preprocessing")

# --- Main Interface ---
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Network Traffic CSV File",
    type=["csv"],
    help="Upload a CSV file containing network flow data"
)

if uploaded_file is not None:
    try:
        # Read CSV
        with st.spinner("📖 Reading CSV file..."):
            df = pd.read_csv(uploaded_file)
        
        
        # Show sample of uploaded data
        with st.expander("🔍 View Uploaded Data (First 5 Rows)"):
            st.dataframe(df.head(), use_container_width=True)
        
        # Store original for display
        display_df = df.copy()
        
        # --- Preprocessing ---
        st.divider()
        st.subheader("⚙️ Preprocessing")
        
        with st.spinner("Processing features..."):
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            df = preprocess_inference(df)
            
            # Get expected feature names from the scaler
            expected_feature_names = scaler.feature_names_in_
            expected_features = len(expected_feature_names)
            actual_features = len(df.columns)
            
            # Check if we have the right number of features
            if actual_features != expected_features:
                st.error(f"""
                ❌ **Feature Mismatch!**
                - Expected: {expected_features} features
                - Got: {actual_features} features
                
                Please ensure your CSV has the same features used during training.
                """)
                st.stop()
            
            # Check for missing columns
            missing_cols = set(expected_feature_names) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_feature_names)
            
            if missing_cols:
                st.error(f"""
                ❌ **Missing columns**: {', '.join(list(missing_cols)[:10])}
                """)
                st.stop()
            
            if extra_cols:
                st.warning(f"⚠️ Extra columns will be ignored: {', '.join(list(extra_cols)[:10])}")
            
            # Reorder columns to match training order
            df = df[expected_feature_names]
            
            st.success(f"✅ Validated {expected_features} features in correct order")
    
            # Scale features
            scaled_features = scaler.transform(df)
            
            # Reshape for Conv1D: (samples, features, 1)
            num_samples = scaled_features.shape[0]
            num_features = scaled_features.shape[1]
            X_input = scaled_features.reshape(num_samples, num_features, 1)
            
            st.success(f"✅ Data preprocessed and ready for inference")
        
        # --- Prediction ---
        st.divider()
        st.subheader("🔮 Making Predictions")
        
        with st.spinner("Running DCNN model..."):
            # Predict
            predictions = model.predict(X_input, verbose=0)
            
            # Get predicted classes (argmax for softmax output)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Get confidence scores
            confidence_scores = np.max(predictions, axis=1)
            
            # Map to labels
            display_df['Prediction'] = [LABEL_MAP[cls] for cls in predicted_classes]
            display_df['Confidence'] = [f"{score:.2%}" for score in confidence_scores]
        
        # --- Results Display ---
        st.divider()
        st.subheader("📊 Results")
        
        # Summary metrics
        n_benign = (predicted_classes == 0).sum()
        n_attack = (predicted_classes == 1).sum()
        attack_percentage = (n_attack / num_samples) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Benign Traffic", n_benign, 
                     delta=f"{(n_benign/num_samples)*100:.1f}%")
        with col2:
            st.metric("🚨 Attacks Detected", n_attack,
                     delta=f"{attack_percentage:.1f}%",
                     delta_color="inverse")
        with col3:
            avg_confidence = confidence_scores.mean()
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        
        # Visualization
        st.subheader("📈 Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            summary_df = pd.DataFrame({
                'Category': ['Benign', 'Attack'],
                'Count': [n_benign, n_attack]
            })
            st.bar_chart(summary_df.set_index('Category'))
        
        with col2:
            # Percentage display
            st.markdown(f"""
            ### Attack Rate
            <div style="font-size: 48px; text-align: center; color: {'red' if attack_percentage > 50 else 'green'};">
                {attack_percentage:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results table
        st.subheader("🔍 Analytics Report")
        
        # Key Statistics
        st.markdown("#### 📊 Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Flows Analyzed", f"{num_samples:,}")
        with col2:
            st.metric("Unique Predictions", len(set(predicted_classes)))
        with col3:
            min_conf = confidence_scores.min()
            st.metric("Min Confidence", f"{min_conf:.2%}")
        with col4:
            max_conf = confidence_scores.max()
            st.metric("Max Confidence", f"{max_conf:.2%}")
        
        # Distribution Analysis
        st.markdown("#### 📈 Prediction Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_counts = display_df['Prediction'].value_counts()
            st.bar_chart(prediction_counts)
        
        with col2:
            # Confidence distribution
            st.markdown("**Confidence Score Distribution**")
            conf_df = pd.DataFrame({
                'Confidence Range': ['90-100%', '80-90%', '70-80%', '60-70%', '<60%'],
                'Count': [
                    ((confidence_scores >= 0.9) & (confidence_scores <= 1.0)).sum(),
                    ((confidence_scores >= 0.8) & (confidence_scores < 0.9)).sum(),
                    ((confidence_scores >= 0.7) & (confidence_scores < 0.8)).sum(),
                    ((confidence_scores >= 0.6) & (confidence_scores < 0.7)).sum(),
                    (confidence_scores < 0.6).sum()
                ]
            })
            st.bar_chart(conf_df.set_index('Confidence Range'))
        
        # Sample of results
        st.markdown("#### 🔬 Sample Results")
        
        tab1, tab2, tab3 = st.tabs(["📋 All Samples", "🚨 Attack Samples", "✅ Benign Samples"])
        
        with tab1:
            st.dataframe(display_df.head(100), use_container_width=True, height=300)
            st.info(f"Showing first 100 of {len(display_df):,} total flows")
        
        with tab2:
            attack_samples = display_df[display_df['Prediction'] == "🚨 ATTACK"]
            if len(attack_samples) > 0:
                st.dataframe(attack_samples.head(100), use_container_width=True, height=300)
                st.info(f"Showing first 100 of {len(attack_samples):,} attack flows")
            else:
                st.success("No attacks detected!")
        
        with tab3:
            benign_samples = display_df[display_df['Prediction'] == "✅ BENIGN"]
            if len(benign_samples) > 0:
                st.dataframe(benign_samples.head(100), use_container_width=True, height=300)
                st.info(f"Showing first 100 of {len(benign_samples):,} benign flows")
            else:
                st.warning("No benign traffic detected!")
        
        # Risk Assessment
        st.markdown("#### ⚠️ Risk Assessment")
        
        if attack_percentage > 75:
            risk_level = "🔴 CRITICAL"
            risk_color = "red"
            risk_msg = "Severe attack detected! Immediate action required."
        elif attack_percentage > 50:
            risk_level = "🟠 HIGH"
            risk_color = "orange"
            risk_msg = "High volume of malicious traffic detected."
        elif attack_percentage > 25:
            risk_level = "🟡 MEDIUM"
            risk_color = "orange"
            risk_msg = "Moderate attack activity detected."
        elif attack_percentage > 5:
            risk_level = "🟢 LOW"
            risk_color = "green"
            risk_msg = "Minor attack activity detected."
        else:
            risk_level = "🟢 MINIMAL"
            risk_color = "green"
            risk_msg = "Network appears secure with minimal threats."
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.1); border-left: 5px solid {risk_color};">
            <h3 style="color: {risk_color};">{risk_level}</h3>
            <p style="font-size: 16px;">{risk_msg}</p>
            <p><strong>Attack Rate:</strong> {attack_percentage:.2f}%</p>
            <p><strong>Average Confidence:</strong> {confidence_scores.mean():.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download results
        st.divider()
        csv_results = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_results,
            file_name="nids_predictions.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        with st.expander("🐛 View Error Details"):
            st.code(traceback.format_exc())

else:
    # Show example data format
    st.info("👆 Upload a CSV file to begin analysis")
    
    with st.expander("📋 Example CSV Format"):
        st.markdown("""
        Your CSV should contain network flow features like:
        - Duration
        - Protocol
        - Flow Bytes/s
        - Flow Packets/s
        - Flow IAT Mean
        - Fwd IAT Mean
        - ... and other CICIDS2017 features
        
        The system will automatically remove identifier columns (Flow ID, IPs, Timestamp).
        """)

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>🛡️ NIDS DCNN v1.0 | Built with Streamlit + TensorFlow</p>
</div>
""", unsafe_allow_html=True)
