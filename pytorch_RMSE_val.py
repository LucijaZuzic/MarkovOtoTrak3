import pandas as pd
import os  
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utilities import load_object
import numpy as np

ws_range = [2, 5, 6, 10, 20, 30]

hidden_range = [220]

model_list = ["GRU", "LSTM", "RNN"]

for varname in os.listdir("final_train_pytorch"):
    
    print(varname)
    
    final_test_NRMSE = []
    final_test_RMSE = []
    final_test_R2 = []
    final_test_MAE = []
    
    hidden_arr = []
    ws_arr = []
    model_arr = []

    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
            
    for model_name in model_list:
        for ws_use in ws_range:
            for hidden_use in hidden_range:
 
                final_test_data = pd.read_csv("final_train_pytorch/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv", sep = ";", index_col = False)
        
                is_a_nan = False
                for val in final_test_data["predicted"]:
                    if str(val) == 'nan':
                        is_a_nan = True
                        break

                hidden_arr.append(hidden_use)
                ws_arr.append(ws_use)
                model_arr.append(model_name)
                    
                if is_a_nan:
                    final_test_MAE.append(1000000)
                    final_test_R2.append(1000000)
                    final_test_NRMSE.append(1000000)
                    final_test_RMSE.append(1000000)
                else:
                    final_test_MAE.append(mean_absolute_error(final_test_data["actual"], final_test_data["predicted"]))
                    final_test_R2.append(r2_score(final_test_data["actual"], final_test_data["predicted"]))
                    final_test_NRMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
                    final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data["predicted"])))
    
    for mini_ix_val in range(len(final_test_RMSE)):
        print(model_arr[mini_ix_val], hidden_arr[mini_ix_val], ws_arr[mini_ix_val], np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))