import pandas as pd
import os  
from utilities import load_object, save_object
import numpy as np
from pytorch_utilities import get_XY
from sacrebleu.metrics import BLEU
 
predicted_all = dict()
y_test_all = dict()
ws_all = dict() 
BLEU_all = dict() 

ws_range = [2, 5, 6, 10, 20, 30]

hidden_range = [100]

model_list = ["GRU", "LSTM", "RNN"]

for varname in os.listdir("final_train_pytorch"):

    predicted_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 
    BLEU_all[varname] = dict() 
 
    for model_name in model_list:
    
        predicted_all[varname][model_name] = dict()
        y_test_all[varname][model_name] = dict()
        ws_all[varname][model_name] = dict() 
        BLEU_all[varname][model_name] = dict() 

        for ws_use in ws_range:
    
            predicted_all[varname][model_name][ws_use] = dict()
            y_test_all[varname][model_name][ws_use] = dict()
            ws_all[varname][model_name][ws_use] = dict() 
            BLEU_all[varname][model_name][ws_use] = dict() 

            for hidden_use in hidden_range:
    
                predicted_all[varname][model_name][ws_use][hidden_use] = dict()
                y_test_all[varname][model_name][ws_use][hidden_use] = dict()
                ws_all[varname][model_name][ws_use][hidden_use] = dict() 
                BLEU_all[varname][model_name][ws_use][hidden_use] = []
  
                final_test_data = pd.read_csv("final_train_pytorch/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv", sep = ";", index_col = False)
    
                file_object_test = load_object("actual/actual_" + varname)
    
                len_total = 0

                for k in file_object_test:
                    
                    ws_all[varname][model_name][ws_use][hidden_use][k] = ws_use

                    x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                    
                    y_test_all[varname][model_name][ws_use][hidden_use][k] = []
                    for ix1 in range(len(y_test_part)): 
                        for ix2 in range(len(y_test_part[ix1])): 
                            y_test_all[varname][model_name][ws_use][hidden_use][k].append(y_test_part[ix1][ix2])
    
                    predicted_all[varname][model_name][ws_use][hidden_use][k] = list(final_test_data["predicted"][len_total:len_total + len(y_test_all[varname][model_name][ws_use][hidden_use][k])])
                    len_total += len(y_test_all[varname][model_name][ws_use][hidden_use][k])   
              
                    bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
                    bleu = BLEU(**bleu_params)
                    pred_str = ""
                    actual_str = "" 
                    round_val = 10
                    if "dir" in varname or "speed" in varname:
                        round_val = 0
                    if "time" in varname:
                        round_val = 3
                    for val_ix in range(len(predicted_all[varname][model_name][ws_use][hidden_use][k])):
                        pred_str += str(np.round(float(predicted_all[varname][model_name][ws_use][hidden_use][k][val_ix]), round_val)) + " "
                        actual_str += str(np.round(float(y_test_all[varname][model_name][ws_use][hidden_use][k][val_ix]), round_val)) + " "
                    pred_str = pred_str[:-1]
                    actual_str = actual_str[:-1]
                    blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
                    BLEU_all[varname][model_name][ws_use][hidden_use].append(blsc)
                    print(varname, model_name, k, BLEU_all[varname][model_name][ws_use][hidden_use][-1])
                    save_object("pytorch_result/BLEU_all", BLEU_all) 
                print(varname, model_name, ws_use, hidden_use, np.mean(BLEU_all[varname][model_name][ws_use][hidden_use]))

for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        for ws_use in BLEU_all[varname][model_name]:

            for hidden_use in BLEU_all[varname][model_name][ws_use]:

                print(varname, model_name, ws_use, hidden_use, np.mean(BLEU_all[varname][model_name][ws_use][hidden_use]))