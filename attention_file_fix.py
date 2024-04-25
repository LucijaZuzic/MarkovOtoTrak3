import os
from pytorch_utilities import fix_file_predictions 

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30]
num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
model_name = "GRU_Att"

for varname in os.listdir("train_attention1"):
    
    print(varname)

    for test_num in range(1, 25):
        ws_use = num_to_ws[test_num] 

        fixfile = "train_attention" + str(test_num) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv"
        if os.path.isfile(fixfile): 
            fix_file_predictions(fixfile)