import pandas as pd
import os
from utilities import load_object
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30]
num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
model_name = "GRU_Att"

for varname in os.listdir("train_attention1"):
    
    print(varname)

    final_test_NRMSE = []
    final_test_RMSE = []
    final_test_R2 = []
    final_test_MAE = []

    test_ix = []
    unk_arr = []
    
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)

    for test_num in range(1, 21):
        ws_use = num_to_ws[test_num] 

        final_test_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv", sep = ";", index_col = False)
        
        final_test_data_predicted = [str(x).strip() for x in final_test_data["predicted"]]
        final_test_data_actual = [str(x).strip() for x in final_test_data["actual"]]

        final_test_data_predicted_new = []

        for ix_x in range(len(final_test_data_predicted)):

            value_ix = final_test_data_predicted[ix_x].replace("a", ".")

            while "  " in value_ix:

                value_ix = value_ix.replace("  ", " ")

            value_ix = value_ix.split(" ")

            while len(value_ix) < ws_use:
                
                value_ix.append(value_ix[-1])

            for vx in range(ws_use):
                
                final_test_data_predicted_new.append(value_ix[vx])

        final_test_data_predicted = final_test_data_predicted_new

        final_test_data_actual_new = []

        for ix_x in range(len(final_test_data_actual)):

            value_ix = final_test_data_actual[ix_x].replace("a", ".")

            while "  " in value_ix:

                value_ix = value_ix.replace("  ", " ")

            for vx in value_ix.split(" "):
                
                final_test_data_actual_new.append(float(vx))

        final_test_data_actual = final_test_data_actual_new
          
        test_unk = 0
        for i in range(len(final_test_data_predicted)):
            if str(final_test_data_predicted[i]) == '<unk>' or str(final_test_data_predicted[i]) == 'n.n':
                test_unk += 1
                if i > 0:
                    final_test_data_predicted[i] = final_test_data_predicted[i - 1]
                else:
                    final_test_data_predicted[i] = 0
            else:
                final_test_data_predicted[i] = float(final_test_data_predicted[i])

        final_test_MAE.append(mean_absolute_error(final_test_data_actual, final_test_data_predicted))
        final_test_R2.append(r2_score(final_test_data_actual, final_test_data_predicted))
        final_test_NRMSE.append(math.sqrt(mean_squared_error(final_test_data_actual, final_test_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))
        final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data_actual, final_test_data_predicted)))

        test_ix.append(test_num)
        unk_arr.append(test_unk / len(final_test_data_predicted))

    for mini_ix_val in range(len(final_test_RMSE)):
        ws_use = num_to_ws[test_ix[mini_ix_val]]
        print(ws_use, num_to_params[test_ix[mini_ix_val]], np.round(unk_arr[mini_ix_val] * 100, 4), np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))