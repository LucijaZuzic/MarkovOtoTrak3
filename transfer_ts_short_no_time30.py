from utilities import load_object
import os
import numpy as np
from datetime import timedelta, datetime
import pandas as pd

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

ws_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19, 20, 25, 29, 30]

for ws_use in ws_range:
    
    yml_part = "task_dataset:"

    yml_part_var = dict()

    for filename in os.listdir("actual_train"):

        if "time" in filename:
            continue

        varname = filename.replace("actual_train_", "")
        
        yml_part_var[varname] = "task_dataset:"

    for filename in os.listdir("actual_train"):

        if "time" in filename:
            continue

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname) 
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        dictio = {"task_name": "pretrain_long_term_forecast", 
                  "dataset": varname, 
                  "data": "custom", 
                  "embed": "timeF", 
                  "root_path": "dataset_new/" + str(ws_use) + "/" + varname, 
                  "data_path": "newdata_TRAIN_short.csv", 
                  "features": "M",
                  "seq_len": ws_use, 
                  "label_len": ws_use, 
                  "pred_len": ws_use, 
                  "enc_in": ws_use, 
                  "dec_in": ws_use, 
                  "c_out": ws_use}

        yml_part += "\n " + str(varname) + ":"
        yml_part_var[varname] += "\n " + str(varname) + ":"
        for v in dictio:
            yml_part += "\n  " + v + ": " + str(dictio[v])
            yml_part_var[varname] += "\n  " + v + ": " + str(dictio[v])
        yml_part += "\n"
        yml_part_var[varname] += "\n"
 
    if not os.path.isdir("csv30data_no_time/data_provider/" + str(ws_use)):
        os.makedirs("csv30data_no_time/data_provider/" + str(ws_use))

    file_yml_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_short.yaml", "w")
    file_yml_pre_write_all.write(yml_part.replace("TRAIN", "ALL"))
    file_yml_pre_write_all.close()

    file_yml_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_short.yaml", "w")
    file_yml_pre_write.write(yml_part)
    file_yml_pre_write.close()
    
    file_yml_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_short.yaml", "w")
    file_yml_pre_write_val.write(yml_part.replace("TRAIN", "TRAIN_VAL"))
    file_yml_pre_write_val.close()
    
    file_yml_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_short.yaml", "w")
    file_yml_write_all.write(yml_part.replace("pretrain_", "").replace("TRAIN", "ALL"))
    file_yml_write_all.close()

    file_yml_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_short.yaml", "w")
    file_yml_write_val.write(yml_part.replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_write_val.close()

    file_yml_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_short.yaml", "w")
    file_yml_write.write(yml_part.replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_write.close()
    
    file_yml_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_short.yaml", "w")
    file_yml_write_train.write(yml_part.replace("pretrain_", ""))
    file_yml_write_train.close()

    file_yml_S_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_S_short.yaml", "w")
    file_yml_S_pre_write_all.write(yml_part.replace("features: M", "features: S").replace("TRAIN", "ALL"))
    file_yml_S_pre_write_all.close()

    file_yml_S_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_S_short.yaml", "w")
    file_yml_S_pre_write.write(yml_part.replace("features: M", "features: S"))
    file_yml_S_pre_write.close()
    
    file_yml_S_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_S_short.yaml", "w")
    file_yml_S_pre_write_val.write(yml_part.replace("features: M", "features: S").replace("TRAIN", "TRAIN_VAL"))
    file_yml_S_pre_write_val.close()
    
    file_yml_S_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_S_short.yaml", "w")
    file_yml_S_write_all.write(yml_part.replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "ALL"))
    file_yml_S_write_all.close()

    file_yml_S_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_S_short.yaml", "w")
    file_yml_S_write_val.write(yml_part.replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_S_write_val.close()

    file_yml_S_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_S_short.yaml", "w")
    file_yml_S_write.write(yml_part.replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_S_write.close()
    
    file_yml_S_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_S_short.yaml", "w")
    file_yml_S_write_train.write(yml_part.replace("features: M", "features: S").replace("pretrain_", ""))
    file_yml_S_write_train.close()

    file_yml_MS_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_MS_short.yaml", "w")
    file_yml_MS_pre_write_all.write(yml_part.replace("features: M", "features: MS").replace("TRAIN", "ALL"))
    file_yml_MS_pre_write_all.close()

    file_yml_MS_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_MS_short.yaml", "w")
    file_yml_MS_pre_write.write(yml_part.replace("features: M", "features: MS"))
    file_yml_MS_pre_write.close()
    
    file_yml_MS_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_MS_short.yaml", "w")
    file_yml_MS_pre_write_val.write(yml_part.replace("features: M", "features: MS").replace("TRAIN", "TRAIN_VAL"))
    file_yml_MS_pre_write_val.close()
    
    file_yml_MS_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_MS_short.yaml", "w")
    file_yml_MS_write_all.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "ALL"))
    file_yml_MS_write_all.close()

    file_yml_MS_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_MS_short.yaml", "w")
    file_yml_MS_write_val.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_MS_write_val.close()

    file_yml_MS_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_MS_short.yaml", "w")
    file_yml_MS_write.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_MS_write.close()
    
    file_yml_MS_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_MS_short.yaml", "w")
    file_yml_MS_write_train.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", ""))
    file_yml_MS_write_train.close()

    for filename in os.listdir("actual_train"):

        if "time" in filename:
            continue

        varname = filename.replace("actual_train_", "")

        file_yml_part_var_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_" + varname + "_short.yaml", "w")
        file_yml_part_var_pre_write_all.write(yml_part_var[varname].replace("TRAIN", "ALL"))
        file_yml_part_var_pre_write_all.close()
 
        file_yml_part_var_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_" + varname + "_short.yaml", "w")
        file_yml_part_var_pre_write.write(yml_part_var[varname])
        file_yml_part_var_pre_write.close()
        
        file_yml_part_var_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_" + varname + "_short.yaml", "w")
        file_yml_part_var_pre_write_val.write(yml_part_var[varname].replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_pre_write_val.close()
        
        file_yml_part_var_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_" + varname + "_short.yaml", "w")
        file_yml_part_var_write_all.write(yml_part_var[varname].replace("pretrain_", "").replace("TRAIN", "ALL"))
        file_yml_part_var_write_all.close()

        file_yml_part_var_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_" + varname + "_short.yaml", "w")
        file_yml_part_var_write_val.write(yml_part_var[varname].replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_write_val.close()

        file_yml_part_var_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_" + varname + "_short.yaml", "w")
        file_yml_part_var_write.write(yml_part_var[varname].replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_write.close()
        
        file_yml_part_var_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_" + varname + "_short.yaml", "w")
        file_yml_part_var_write_train.write(yml_part_var[varname].replace("pretrain_", ""))
        file_yml_part_var_write_train.close()

        file_yml_part_var_S_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_pre_write_all.write(yml_part_var[varname].replace("features: M", "features: S").replace("TRAIN", "ALL"))
        file_yml_part_var_S_pre_write_all.close()

        file_yml_part_var_S_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_pre_write.write(yml_part_var[varname].replace("features: M", "features: S"))
        file_yml_part_var_S_pre_write.close()
        
        file_yml_part_var_S_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_pre_write_val.write(yml_part_var[varname].replace("features: M", "features: S").replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_S_pre_write_val.close()
        
        file_yml_part_var_S_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_write_all.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "ALL"))
        file_yml_part_var_S_write_all.close()

        file_yml_part_var_S_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_write_val.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_S_write_val.close()

        file_yml_part_var_S_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_write.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_S_write.close()
        
        file_yml_part_var_S_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_S_" + varname + "_short.yaml", "w")
        file_yml_part_var_S_write_train.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", ""))
        file_yml_part_var_S_write_train.close()

        file_yml_part_var_MS_pre_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_all_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_pre_write_all.write(yml_part_var[varname].replace("features: M", "features: MS").replace("TRAIN", "ALL"))
        file_yml_part_var_MS_pre_write_all.close()

        file_yml_part_var_MS_pre_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_pre_write.write(yml_part_var[varname].replace("features: M", "features: MS"))
        file_yml_part_var_MS_pre_write.close()
        
        file_yml_part_var_MS_pre_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30multi_task_pretrain_no_time_val_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_pre_write_val.write(yml_part_var[varname].replace("features: M", "features: MS").replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_MS_pre_write_val.close()
        
        file_yml_part_var_MS_write_all = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_all_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_write_all.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "ALL"))
        file_yml_part_var_MS_write_all.close()

        file_yml_part_var_MS_write_val = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_val_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_write_val.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_MS_write_val.close()

        file_yml_part_var_MS_write = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_write.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_MS_write.close()
        
        file_yml_part_var_MS_write_train = open("csv30data_no_time/data_provider/" + str(ws_use) + "/30zeroshot_task_no_time_train_MS_" + varname + "_short.yaml", "w")
        file_yml_part_var_MS_write_train.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", ""))
        file_yml_part_var_MS_write_train.close()