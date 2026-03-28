# Anomaly Detection in Nanosensor Data

This project implements an anomaly detection system for nanosensor IoT data using a **PyTorch-based LSTM Autoencoder**. The model learns the normal behavior of sensor readings and identifies anomalies when the reconstruction error (MSE) exceeds a certain threshold.

## Features
- LSTM Autoencoder model built with PyTorch
- Detection of anomalies using reconstruction error (MSE)
- Simulation of IoT sensor data stream
- Artificial anomaly injection for testing
- Training loss visualization

## Dataset
The dataset contains nanosensor readings with the following features:
- Temperature
- Humidity
- Air Quality
- Light
- Loudness

Dataset file: `dataset_final.csv`

## How It Works
1. Load and preprocess nanosensor data.
2. Normalize the features using MinMaxScaler.
3. Create time-series sequences.
4. Train an LSTM Autoencoder model.
5. Calculate reconstruction error (MSE).
6. Detect anomalies when error exceeds a threshold.

## Project Structure
project-folder
│
├── dataset_final.csv
├── train_lstm_autoencoder.py
├── run.py
├── models
├── logs
├── plots
└── README.md


## Installation
Install required libraries:
pip install torch pandas numpy scikit-learn matplotlib joblib

## Training the Model
Run the training script:
python train_lstm_autoencoder.py

## Running the Data Simulation
To simulate IoT nanosensor data with injected anomalies:
python run.py

## Technologies Used
- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Applications
- IoT monitoring systems
- Environmental monitoring
- Industrial anomaly detection
- Smart sensor networks
