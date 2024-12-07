# Time Series Forecasting with RNN, LSTM, and GRU

This repository demonstrates the use of Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models to predict future values of a time series. In this example, we forecast daily minimum temperatures using data from the [UCI Machine Learning Repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv).

## Project Overview

The goal of this project is to apply and compare different types of Recurrent Neural Networks (RNNs) for time series forecasting. Specifically, we focus on the following models:
- **RNN (Recurrent Neural Network)**
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

The model is trained on a dataset containing daily minimum temperatures, and the task is to predict the temperature for the next day based on the past 10 days.

## Requirements

The project requires the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (for building and training the RNN, LSTM, and GRU models)

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
Files in this Repository
time_series_forecasting.py: Contains the main implementation for loading, processing, and training the models.
README.md: This file, which provides an overview of the project.
requirements.txt: A file listing the dependencies for the project.
Steps to Run the Project
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/time-series-forecasting.git
cd time-series-forecasting
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the script to train and evaluate the models:

bash
Copy code
python time_series_forecasting.py
The script will:

Load the daily minimum temperatures dataset.
Normalize the data to scale the temperature values between 0 and 1.
Create sequences from the data for supervised learning.
Train three models: RNN, LSTM, and GRU.
Evaluate each model's performance on the test set.
Plot the training and validation loss for each model.
Key Concepts
Time Series Forecasting: Predicting future values based on historical data. This project uses daily minimum temperature data.
Recurrent Neural Networks (RNNs): A class of neural networks designed for sequence prediction, where the output of a previous step is used as input for the next step.
Long Short-Term Memory (LSTM): A type of RNN that is designed to handle long-range dependencies by mitigating the vanishing gradient problem.
Gated Recurrent Unit (GRU): A variant of LSTM with a simpler structure, often providing faster training times while still capturing long-term dependencies.
Results
The performance of the models is evaluated based on the mean squared error (MSE) on the test data. The training and validation losses for each model are plotted to visualize their learning process.

Future Work
Experiment with more advanced architectures (e.g., Bidirectional RNNs or Attention Mechanisms).
Tune hyperparameters (e.g., number of units, learning rate) for better model performance.
Apply the models to different time series datasets (e.g., stock prices, energy consumption).
