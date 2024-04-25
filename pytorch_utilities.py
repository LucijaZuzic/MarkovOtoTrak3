import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def fix_file_predictions(name_file):
    with open(name_file,'r') as f:
        lines = f.readlines()
        strpr = {"actual": [], "predicted": []}
        cum_c = 0
        buffer = ''
        for line in lines:
            buffer += line # Append the current line to a buffer
            cum_c = buffer.count(';')
            if cum_c == 1:
                if "actual" not in buffer:
                    strpr["actual"].append(buffer.split(";")[0].replace("\n", "").replace('"', ""))
                    strpr["predicted"].append(buffer.split(";")[1].replace("\n", "").replace('"', ""))
                buffer = ''
            elif cum_c > 1:
                raise # This should never happen
        df_new = pd.DataFrame(strpr)
        df_new.to_csv(name_file, index = False, sep = ";")

def print_predictions(actual, predicted, name_file):
    
    strpr = {"actual": [], "predicted": []}
    for ix1 in range(len(actual)):
        for ix2 in range(len(actual[ix1])):
            strpr["actual"].append(str(actual[ix1][ix2]).replace("[", "").replace("]", "").replace("tensor(", "").replace(")", ""))
            strpr["predicted"].append(str(predicted[ix1][ix2]).replace("[", "").replace("]", "").replace("tensor(", "").replace(")", ""))
    df_new = pd.DataFrame(strpr)
    df_new.to_csv(name_file, index = False, sep = ";") 

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

class PyTorchGRUModel(nn.Module):
    def __init__(self, input_size, hidden_units, dense_units):
        super(PyTorchGRUModel, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units, batch_first=True)

        # Dense layer with linear activation
        self.fc = nn.Linear(hidden_units, dense_units)

    def forward(self, x):
        # GRU layer
        gru_output, _ = self.gru(x)

        # Extract the last hidden state from the GRU output
        last_hidden_state = gru_output[:, :]

        # Dense layer with linear activation
        output = self.fc(last_hidden_state)

        return output
    
class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, dense_units):
        super(PyTorchLSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units, batch_first=True)

        # Dense layer with linear activation
        self.fc = nn.Linear(hidden_units, dense_units)

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.lstm(x)

        # Extract the last hidden state from the LSTM output
        last_hidden_state = lstm_output[:, :]

        # Dense layer with linear activation
        output = self.fc(last_hidden_state)

        return output
    
class PyTorchRNNModel(nn.Module):
    def __init__(self, input_size, hidden_units, dense_units):
        super(PyTorchRNNModel, self).__init__()

        # SimpleRNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units, batch_first=True)

        # Dense layer with linear activation
        self.fc = nn.Linear(hidden_units, dense_units)

    def forward(self, x):
        # SimpleRNN layer
        rnn_output, _ = self.rnn(x)

        # Extract the last hidden state from the SimpleRNN output
        last_hidden_state = rnn_output[:, :]

        # Dense layer with linear activation
        output = self.fc(last_hidden_state)

        return output