from utilities import load_object
import numpy as np

BLEU_all = load_object("attention_result/BLEU_all")

for varname in BLEU_all:
    
    for test_num in BLEU_all[varname]:

        for model_name in BLEU_all[varname][test_num]:

            print(varname, test_num, model_name, np.mean(BLEU_all[varname][test_num][model_name]))