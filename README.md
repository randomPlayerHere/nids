# Network Intrusion Detection System 🛡️

A machine learning-based Network Intrusion Detection System (NIDS) built with scikit-learn and Streamlit. The system uses KDD Cup dataset to detect network attacks in real-time through an interactive web interface.

## Features

- **Machine Learning Pipeline**: KNN classifier with PCA dimensionality reduction
- **Interactive UI**: Streamlit-based web interface for real-time predictions
- **Multiple Prediction Modes**:
  - Single connection analysis
  - Batch prediction
  - Random sample testing
- **Performance Metrics**: Real-time accuracy, confidence scores, and prediction distributions
- **Data Visualization**: Connection details and prediction distributions

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/randomPlayerHere/nids.git
cd nids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (if not using pre-trained):
```bash
python train.py
```

4. Launch the web interface:
```bash
streamlit run app.py
```

## Dataset

The project uses the KDD Cup dataset for network intrusion detection. The dataset includes various network connection features and labels indicating normal traffic or different types of attacks.

Key features include:
- Protocol type
- Service
- Connection statistics
- Traffic features
- Content features

## Project Structure

```
nids/
├── app.py              # Streamlit web application
├── train.py            # Model training script
├── data/              
│   ├── KDDTrain+.csv   # Training data
│   └── KDDTest+.csv    # Test data
├── models/             # Saved model files
│   ├── knn_pipeline.pkl
│   └── ...
└── requirements.txt    # Project dependencies
```

## Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Feature Engineering**: 
  - PCA (20 components)
  - Robust scaling for numerical features
  - One-hot encoding for categorical features
- **Binary Classification**: Normal vs Attack traffic
- **Preprocessing Pipeline**: Automated data preprocessing using scikit-learn pipelines

## Usage Examples

### Single Connection Analysis
Analyze individual network connections:
1. Select "Single Prediction" mode
2. Enter the connection index
3. Click "Predict" to see detailed analysis

### Batch Processing
Process multiple connections at once:
1. Choose "Batch Prediction"
2. Select number of samples
3. View aggregate statistics and individual results

### Random Sampling
Test random connections:
1. Select "Random Sample" mode
2. Choose sample size
3. Analyze distribution of predictions

## Performance

The model achieves:
- High accuracy in detecting network intrusions
- Real-time prediction capabilities
- Robust feature preprocessing
- Confidence scoring for predictions

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- streamlit
- joblib

## Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues or fork the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.