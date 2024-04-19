from utilities import load_object
import numpy as np

BLEU_all = load_object("attention_result/BLEU_all")

for varname in BLEU_all:
    
    for test_num in BLEU_all[varname]:

        for model_name in BLEU_all[varname][test_num]:

            print(varname, test_num, model_name, np.mean(BLEU_all[varname][test_num][model_name]))

BLEU_all = load_object("UniTS_final_result/BLEU_all")
            
for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        for ws_use in BLEU_all[varname][model_name]:

            print(varname, model_name, ws_use, np.mean(BLEU_all[varname][model_name][ws_use]))

BLEU_all = load_object("pytorch_result/BLEU_all")
            
for varname in BLEU_all:
    
    for model_name in BLEU_all[varname]:

        for ws_use in BLEU_all[varname][model_name]:

            for hidden_use in BLEU_all[varname][model_name][ws_use]:

                print(varname, model_name, ws_use, hidden_use, np.mean(BLEU_all[varname][model_name][ws_use][hidden_use]))