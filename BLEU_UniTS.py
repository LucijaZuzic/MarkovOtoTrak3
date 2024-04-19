import pandas as pd
import os  
from utilities import load_object, save_object
from sacrebleu.metrics import BLEU
import numpy as np

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

predicted_all = dict() 
y_test_all = dict()
ws_all = dict() 
BLEU_all = dict()
ws_range = range(5, 7)
model_name = "UniTS"

for varname in os.listdir("final_train_pytorch"):

    predicted_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 
    BLEU_all[varname] = dict()

    predicted_all[varname][model_name] = dict()
    y_test_all[varname][model_name] = dict()
    ws_all[varname][model_name] = dict() 
    BLEU_all[varname][model_name] = dict() 
    
    for ws_use in ws_range:

        predicted_all[varname][model_name][ws_use] = dict()
        y_test_all[varname][model_name][ws_use] = dict()
        ws_all[varname][model_name][ws_use] = dict() 
        BLEU_all[varname][model_name][ws_use] = []
  
        final_test_data = pd.read_csv("UniTS_final_res/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

        file_object_test = load_object("actual/actual_" + varname)

        len_total = 0

        for k in file_object_test:
            
            ws_all[varname][model_name][ws_use][k] = ws_use
            
            x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, 1)
            
            y_test_all[varname][model_name][ws_use][k] = []
            for ix1 in range(len(y_test_part)): 
                for ix2 in range(len(y_test_part[ix1])): 
                    y_test_all[varname][model_name][ws_use][k].append(y_test_part[ix1][ix2])

            predicted_all[varname][model_name][ws_use][k] = list(final_test_data["predicted"][len_total:len_total + len(y_test_all[varname][model_name][ws_use][k])])
            len_total += len(y_test_all[varname][model_name][ws_use][k]) 
              
            bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
            bleu = BLEU(**bleu_params)
            pred_str = ""
            actual_str = "" 
            for val_ix in range(len(predicted_all[varname][model_name][ws_use][k])):
                pred_str += str(predicted_all[varname][model_name][ws_use][k]) + " "
                actual_str += str(y_test_all[varname][model_name][ws_use][k]) + " "
            pred_str = pred_str[:-1]
            actual_str = actual_str[:-1] 
            blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
            BLEU_all[varname][model_name][ws_use].append(blsc)
            print(varname, model_name, k, BLEU_all[varname][model_name][ws_use][-1])
            save_object("UniTS_final_result/BLEU_all", BLEU_all) 
        print(varname, model_name, ws_use, np.mean(BLEU_all[varname][model_name][ws_use]))

for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        for ws_use in BLEU_all[varname][model_name]:

            print(varname, model_name, ws_use, np.mean(BLEU_all[varname][model_name][ws_use]))