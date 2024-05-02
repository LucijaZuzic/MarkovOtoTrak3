import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import load_object
from pytorch_utilities import get_XY, print_predictions, PyTorchGRUModel, PyTorchLSTMModel, PyTorchRNNModel

num_props = 1

ws_range = [15, 19, 25, 29]

hidden_range = [100]

model_list = ["GRU", "LSTM", "RNN"]

for filename in os.listdir("actual_train"):

    varname = filename.replace("actual_train_", "")

    file_object_train_val = load_object("actual_train_val/actual_train_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)

    for model_name in model_list:

        for ws_use in ws_range:
            
            x_train_val_all = []
            y_train_val_all = []

            for k in file_object_train_val:

                x_train_val_part, y_train_val_part = get_XY(file_object_train_val[k], ws_use, 1, ws_use)
                
                for ix in range(len(x_train_val_part)):
                    x_train_val_all.append(x_train_val_part[ix]) 
                    y_train_val_all.append(y_train_val_part[ix])

            x_train_val_all = np.array(x_train_val_all)
            y_train_val_all = np.array(y_train_val_all)
            
            x_test_all = []
            y_test_all = []

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                
                for ix in range(len(x_test_part)):
                    x_test_all.append(x_test_part[ix]) 
                    y_test_all.append(y_test_part[ix])

            x_test_all = np.array(x_test_all)
            y_test_all = np.array(y_test_all)
 
            for hidden_use in hidden_range:

                if model_name == "RNN":
                    pytorch_model = PyTorchRNNModel(ws_use, hidden_use, ws_use)

                if model_name == "GRU": 
                    pytorch_model = PyTorchGRUModel(ws_use, hidden_use, ws_use)

                if model_name == "LSTM": 
                    pytorch_model = PyTorchLSTMModel(ws_use, hidden_use, ws_use)

                if not os.path.isdir("final_train_pytorch/" + varname + "/models/" + model_name):
                    os.makedirs("final_train_pytorch/" + varname + "/models/" + model_name)
            
                if not os.path.isdir("final_train_pytorch/" + varname + "/predictions/test/" + model_name):
                    os.makedirs("final_train_pytorch/" + varname + "/predictions/test/" + model_name)
            
                train_dataset = TensorDataset(torch.tensor(x_train_val_all).float(),  torch.tensor(y_train_val_all).float())
                train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(pytorch_model.parameters())

                num_epochs = 20
                for epoch in range(num_epochs):
                    for inputs, targets in train_loader:
                        pytorch_model.train()
                        optimizer.zero_grad()
                        outputs = pytorch_model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                pytorch_model.eval()

                torch.save(pytorch_model.state_dict(), "final_train_pytorch/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".pth")

                with torch.no_grad():

                    predict_test_all = pytorch_model(torch.tensor(x_test_all).float())
                    
                    print_predictions(y_test_all, predict_test_all, "final_train_pytorch/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")