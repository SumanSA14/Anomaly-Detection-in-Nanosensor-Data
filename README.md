Anomaly Detection in Nanosensor Data using LSTM Autoencoder
Overview

This project implements an anomaly detection system for nanosensor-based IoT data streams using a deep learning LSTM Autoencoder built with PyTorch. The model learns the normal behavior of sensor readings and detects anomalies by measuring the reconstruction error using Mean Squared Error (MSE).

The system simulates a real-time IoT sensor stream and periodically injects artificial anomalies to evaluate how effectively the model identifies abnormal patterns.

This project demonstrates how deep learning models can monitor sensor networks and detect abnormal system behavior automatically.

Features
Deep learning LSTM Autoencoder for anomaly detection
Real-time IoT sensor data simulation
Automatic anomaly injection for testing
Detection based on reconstruction error (MSE)
Training loss logging and visualization
Model checkpoint saving
Scaler persistence for consistent preprocessing
Visualization of training performance
Project Architecture
Sensor Data (CSV Dataset)
        │
        ▼
Data Preprocessing
(MinMax Scaling)
        │
        ▼
Sequence Creation
(Time-series windows)
        │
        ▼
LSTM Autoencoder Model
        │
        ▼
Reconstruction of Input Sequence
        │
        ▼
Reconstruction Error (MSE)
        │
        ▼
Threshold Detection
        │
        ▼
Anomaly / Normal Classification
Dataset

The dataset contains simulated nanosensor readings with the following features:

Sensor	Description
Temperature	Environmental temperature reading
Humidity	Air humidity level
Air Quality	Pollution / air quality measurement
Light	Ambient light intensity
Loudness	Environmental sound level

The dataset is stored as:

dataset_final.csv
Data Simulation

The system simulates a live IoT data stream and injects anomalies periodically to test detection capability.

Artificial anomalies are generated every 15 seconds by modifying sensor values such as temperature, humidity, and light intensity.

Example simulation output:

Timestamp                Status     Data
2026-03-20 12:00:01      NORMAL     22.1 | 45.2 | 30.4 | 200 | 60
2026-03-20 12:00:15      ANOMALY    30.8 | 65.4 | 30.2 | 420 | 61

This behavior is implemented in the simulation script.

Model Architecture

The anomaly detection model is based on a Long Short-Term Memory (LSTM) Autoencoder.

The encoder compresses the input sequence into a hidden representation, while the decoder reconstructs the original sequence.

Input Sequence
     │
     ▼
LSTM Encoder
     │
     ▼
Latent Representation
     │
     ▼
LSTM Decoder
     │
     ▼
Reconstructed Sequence

If the reconstruction error exceeds a threshold, the sequence is classified as an anomaly.

The implementation of the LSTM Autoencoder model is provided in the training script.

Model Training
Hyperparameters
Parameter	Value
Sequence Length	10
Batch Size	32
Epochs	50
Hidden Size	64
Learning Rate	0.001

The training pipeline includes:

Loading sensor data
Scaling features using MinMaxScaler
Creating time-series sequences
Training the LSTM Autoencoder
Computing reconstruction loss (MSE)
Saving the trained model and scaler
Training Loss Visualization

The training script automatically generates a loss curve showing how the model improves over time.

Saved file:

plots/loss_curve.png

This plot helps analyze the model's learning progress and convergence.

Project Structure
Anomaly-Detection-Nanosensor
│
├── dataset_final.csv
├── train_lstm_autoencoder.py
├── run.py
│
├── models
│   ├── lstm_autoencoder.pt
│   └── scaler.pkl
│
├── logs
│   └── training_metrics.csv
│
├── plots
│   └── loss_curve.png
│
└── README.md
Installation

Clone the repository

git clone https://github.com/yourusername/anomaly-detection-nanosensor.git
cd anomaly-detection-nanosensor

Install dependencies

pip install torch pandas numpy scikit-learn matplotlib joblib
Training the Model

Run the training script:

python train_lstm_autoencoder.py

This will:

Train the LSTM Autoencoder
Save the model
Save the scaler
Generate the training loss plot
Log training metrics
Running the Data Stream Simulation

To simulate IoT sensor data with artificial anomalies:

python run.py

This will generate a live sensor data stream and print whether the data is NORMAL or ANOMALOUS.

Results

The LSTM Autoencoder learns the normal patterns of nanosensor data and detects anomalies when reconstruction error increases significantly.

Key observations:

The model successfully learns normal sensor behavior
Artificial anomalies produce higher reconstruction error
The system can be extended for real-time IoT monitoring
Future Improvements

Possible enhancements for this project:

Real-time anomaly detection dashboard
Automatic threshold tuning
Streaming data pipeline integration
Transformer-based anomaly detection models
Deployment using FastAPI or Streamlit
Visualization dashboard for anomaly monitoring
Technologies Used
Python
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
Applications

This type of anomaly detection system can be used in:

Smart cities
Environmental monitoring
Industrial IoT systems
Predictive maintenance
Sensor network monitoring
