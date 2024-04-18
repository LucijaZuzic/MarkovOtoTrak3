import pandas as pd
import os  
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utilities import load_object
import numpy as np

ws_range = range(4, 7)
 
for varname in os.listdir("final_train_pytorch"):
    
    print(varname)
    
    final_test_NRMSE = []
    final_test_RMSE = []
    final_test_R2 = []
    final_test_MAE = []
     
    ws_arr = []
    model_arr = []

    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
             
    for ws_use in ws_range: 

        final_test_data = pd.read_csv("UniTS_final_res/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

        is_a_nan = False
        for val in final_test_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break

        ws_arr.append(ws_use) 
            
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
        print(ws_arr[mini_ix_val], np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))