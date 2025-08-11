# Google Stock Price Prediction

This project aims to predict the closing price of Google's stock (GOOGL) using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models. The models are built from scratch using PyTorch.

## Repository Structure

```
google-stock-prediction/
├── data/
│   └── GOOGL.csv         # Google stock data
├── notebooks/
│   ├── lstm.ipynb        # Jupyter notebook for the LSTM model
│   └── rnn.ipynb         # Jupyter notebook for the RNN model
└── README.md             # This file
```

## Dataset

The dataset used is `GOOGL.csv`, which contains historical stock data for Google. The following features are used for training the models:
- **Open**: The price at which the stock first traded upon the opening of an exchange on a given trading day.
- **High**: The highest price at which a stock traded during the course of a trading day.
- **Low**: The lowest price at which a stock traded during the course of a trading day.
- **Close**: The last price at which a stock trades during a regular trading session.

The goal is to predict the `Close` price of the next day based on the previous 30 days of data.

## Models

Two types of recurrent neural networks are implemented in this project:

1.  **Simple RNN [rnn.ipynb](/notebooks/rnn.ipynb)**: A basic Recurrent Neural Network built from scratch.
2.  **LSTM [lstm.ipynb](/notebooks/lstm.ipynb)**: A Long Short-Term Memory network, also built from scratch, which is generally better at capturing long-term dependencies in sequential data.

Both models are implemented using PyTorch.

### Methodology

1.  **Data Loading**: The `GOOGL.csv` dataset is loaded using the pandas library.
2.  **Preprocessing**:
    - The `Open`, `High`, `Low`, and `Close` features are selected.
    - The data is normalized to a range of (0, 1) using `MinMaxScaler` from scikit-learn.
3.  **Sequence Creation**: The data is transformed into sequences. Each sequence consists of 30 days of data (`[Open, High, Low, Close]`) and the target is the `Close` price of the 31st day.
4.  **Data Splitting**: The dataset is split into a training set (80%) and a testing set (20%).
5.  **Model Training**:
    - The models are trained for 10 epochs.
    - **Loss Function**: Mean Squared Error (`nn.MSELoss`).
    - **Optimizer**: Adam optimizer with a learning rate of `0.0001`.
    - **Batch Size**: 32.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd google-stock-prediction
    ```

2.  **Install dependencies:**
    Make sure you have Python and Jupyter Notebook installed. You can install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib scikit-learn torch
    ```
    If you have a CUDA-enabled GPU, you can install the GPU version of PyTorch for faster training.

3.  **Run the notebooks:**
    Launch Jupyter Notebook and open either `notebooks/rnn.ipynb` or `notebooks/lstm.ipynb`.
    ```bash
    jupyter notebook
    ```
    You can then run the cells in the notebook to train the model and see the results.

## Results

asdBoth models perform well on the dataset with really small test losses (about 0.0002). Both models are trained, and their training and testing losses are plotted over the epochs. The plots can be found at the end of each notebook. The losses decrease significantly after the first epoch, indicating that the models are learning to predict the stock prices effectively.

For detailed implementation and results, please refer to the Jupyter notebooks.
