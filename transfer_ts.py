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

ws_range = [9, 19, 29]

for ws_use in ws_range:
    
    yml_part = "task_dataset:"

    yml_part_var = dict()

    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")
        
        yml_part_var[varname] = "task_dataset:"

    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname) 
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        dictio = {"task_name": "pretrain_long_term_forecast", 
                  "dataset": varname, 
                  "data": "custom", 
                  "embed": "timeF", 
                  "root_path": "dataset_new/" + str(ws_use) + "/" + varname, 
                  "data_path": "newdata_TRAIN.csv", 
                  "features": "M",
                  "seq_len": ws_use, 
                  "label_len": 0, 
                  "pred_len": 1, 
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
         
        str_train = {"date": []}
        str_val = {"date": []}
        str_test = {"date": []}
        str_train_val = {"date": []}
        str_all = {"date": []}

        datetime_use = datetime(day = 1, month = 1, year = 1970)

        for ik in range(ws_use):
            str_train[str(ik)] = []
            str_val[str(ik)] = []
            str_test[str(ik)] = []
            str_train_val[str(ik)] = []
            str_all[str(ik)] = []

        str_train["OT"] = []
        str_val["OT"] = []
        str_test["OT"] = []
        str_train_val["OT"] = []
        str_all["OT"] = []
        
        for k in file_object_train:

            x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, 1, 1)
            
            for ix1 in range(len(x_train_part)):
                    
                str_train["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))
                str_train_val["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))
                str_all["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))

                for ix2 in range(len(x_train_part[ix1])): 
                    str_train[str(ix2)].append(str(x_train_part[ix1][ix2]).replace(",", "."))
                    str_train_val[str(ix2)].append(str(x_train_part[ix1][ix2]).replace(",", ".")) 
                    str_all[str(ix2)].append(str(x_train_part[ix1][ix2]).replace(",", ".")) 
                    
                for ix2 in range(len(y_train_part[ix1])): 
                    str_train["OT"].append(str(y_train_part[ix1][ix2]).replace(",", "."))
                    str_train_val["OT"].append(str(y_train_part[ix1][ix2]).replace(",", "."))
                    str_all["OT"].append(str(y_train_part[ix1][ix2]).replace(",", "."))
                
                datetime_use += timedelta(hours = 1)

        for k in file_object_val:

            x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, 1, 1)
            
            for ix1 in range(len(x_val_part)):

                str_val["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))
                str_train_val["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))
                str_all["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))

                for ix2 in range(len(x_val_part[ix1])):
                    str_val[str(ix2)].append(str(x_val_part[ix1][ix2]).replace(",", "."))
                    str_train_val[str(ix2)].append(str(x_val_part[ix1][ix2]).replace(",", "."))
                    str_all[str(ix2)].append(str(x_val_part[ix1][ix2]).replace(",", ".")) 

                for ix2 in range(len(y_val_part[ix1])):
                    str_val["OT"].append(str(y_val_part[ix1][ix2]).replace(",", "."))
                    str_train_val["OT"].append(str(y_val_part[ix1][ix2]).replace(",", "."))
                    str_all["OT"].append(str(y_val_part[ix1][ix2]).replace(",", "."))
                
                datetime_use += timedelta(hours = 1)

        for k in file_object_test:

            x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, 1)
            
            for ix1 in range(len(x_test_part)):

                str_test["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))
                str_all["date"].append(datetime.strftime(datetime_use, "%Y-%m-%d %H-%M-%S"))

                for ix2 in range(len(x_test_part[ix1])):
                    str_test[str(ix2)].append(str(x_test_part[ix1][ix2]).replace(",", "."))
                    str_all[str(ix2)].append(str(x_test_part[ix1][ix2]).replace(",", ".")) 

                for ix2 in range(len(y_test_part[ix1])):
                    str_test["OT"].append(str(y_test_part[ix1][ix2]).replace(",", "."))
                    str_all["OT"].append(str(y_test_part[ix1][ix2]).replace(",", "."))
                    
                datetime_use += timedelta(hours = 1)

        if not os.path.isdir("csv_data/dataset/" + str(ws_use) + "/" + varname):
            os.makedirs("csv_data/dataset/" + str(ws_use) + "/" + varname)

        file_train_write = pd.DataFrame(str_train) 
        file_train_write.to_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TRAIN.csv", index = False, sep = ",") 

        file_val_write = pd.DataFrame(str_val) 
        file_val_write.to_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_VAL.csv", index = False, sep = ",") 

        file_train_val_write = pd.DataFrame(str_train_val) 
        file_train_val_write.to_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TRAIN_VAL.csv", index = False, sep = ",") 

        file_test_write = pd.DataFrame(str_all) 
        file_test_write.to_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TEST.csv", index = False, sep = ",") 

        file_all_write = pd.DataFrame(str_test) 
        file_all_write.to_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_ALL.csv", index = False, sep = ",") 
    
    if not os.path.isdir("csv_data/data_provider/" + str(ws_use)):
        os.makedirs("csv_data/data_provider/" + str(ws_use))

    file_yml_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain.yaml", "w")
    file_yml_pre_write.write(yml_part)
    file_yml_pre_write.close()
    
    file_yml_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val.yaml", "w")
    file_yml_pre_write_val.write(yml_part.replace("TRAIN", "TRAIN_VAL"))
    file_yml_pre_write_val.close()
    
    file_yml_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val.yaml", "w")
    file_yml_write_val.write(yml_part.replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_write_val.close()

    file_yml_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task.yaml", "w")
    file_yml_write.write(yml_part.replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_write.close()
    
    file_yml_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train.yaml", "w")
    file_yml_write_train.write(yml_part.replace("pretrain_", ""))
    file_yml_write_train.close()

    file_yml_S_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_S.yaml", "w")
    file_yml_S_pre_write.write(yml_part.replace("features: M", "features: S"))
    file_yml_S_pre_write.close()
    
    file_yml_S_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val_S.yaml", "w")
    file_yml_S_pre_write_val.write(yml_part.replace("features: M", "features: S").replace("TRAIN", "TRAIN_VAL"))
    file_yml_S_pre_write_val.close()
    
    file_yml_S_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val_S.yaml", "w")
    file_yml_S_write_val.write(yml_part.replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_S_write_val.close()

    file_yml_S_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_S.yaml", "w")
    file_yml_S_write.write(yml_part.replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_S_write.close()
    
    file_yml_S_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train_S.yaml", "w")
    file_yml_S_write_train.write(yml_part.replace("features: M", "features: S").replace("pretrain_", ""))
    file_yml_S_write_train.close()

    file_yml_MS_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_MS.yaml", "w")
    file_yml_MS_pre_write.write(yml_part.replace("features: M", "features: MS"))
    file_yml_MS_pre_write.close()
    
    file_yml_MS_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val_MS.yaml", "w")
    file_yml_MS_pre_write_val.write(yml_part.replace("features: M", "features: MS").replace("TRAIN", "TRAIN_VAL"))
    file_yml_MS_pre_write_val.close()
    
    file_yml_MS_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val_MS.yaml", "w")
    file_yml_MS_write_val.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "VAL"))
    file_yml_MS_write_val.close()

    file_yml_MS_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_MS.yaml", "w")
    file_yml_MS_write.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "TEST"))
    file_yml_MS_write.close()
    
    file_yml_MS_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train_MS.yaml", "w")
    file_yml_MS_write_train.write(yml_part.replace("features: M", "features: MS").replace("pretrain_", ""))
    file_yml_MS_write_train.close()

    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")
 
        file_yml_part_var_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_" + varname + ".yaml", "w")
        file_yml_part_var_pre_write.write(yml_part_var[varname])
        file_yml_part_var_pre_write.close()
        
        file_yml_part_var_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val_" + varname + ".yaml", "w")
        file_yml_part_var_pre_write_val.write(yml_part_var[varname].replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_pre_write_val.close()
        
        file_yml_part_var_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val_" + varname + ".yaml", "w")
        file_yml_part_var_write_val.write(yml_part_var[varname].replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_write_val.close()

        file_yml_part_var_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_" + varname + ".yaml", "w")
        file_yml_part_var_write.write(yml_part_var[varname].replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_write.close()
        
        file_yml_part_var_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train_" + varname + ".yaml", "w")
        file_yml_part_var_write_train.write(yml_part_var[varname].replace("pretrain_", ""))
        file_yml_part_var_write_train.close()

        file_yml_part_var_S_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_S_" + varname + ".yaml", "w")
        file_yml_part_var_S_pre_write.write(yml_part_var[varname].replace("features: M", "features: S"))
        file_yml_part_var_S_pre_write.close()
        
        file_yml_part_var_S_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val_S_" + varname + ".yaml", "w")
        file_yml_part_var_S_pre_write_val.write(yml_part_var[varname].replace("features: M", "features: S").replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_S_pre_write_val.close()
        
        file_yml_part_var_S_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val_S_" + varname + ".yaml", "w")
        file_yml_part_var_S_write_val.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_S_write_val.close()

        file_yml_part_var_S_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_S_" + varname + ".yaml", "w")
        file_yml_part_var_S_write.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_S_write.close()
        
        file_yml_part_var_S_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train_S_" + varname + ".yaml", "w")
        file_yml_part_var_S_write_train.write(yml_part_var[varname].replace("features: M", "features: S").replace("pretrain_", ""))
        file_yml_part_var_S_write_train.close()

        file_yml_part_var_MS_pre_write = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_MS_" + varname + ".yaml", "w")
        file_yml_part_var_MS_pre_write.write(yml_part_var[varname].replace("features: M", "features: MS"))
        file_yml_part_var_MS_pre_write.close()
        
        file_yml_part_var_MS_pre_write_val = open("csv_data/data_provider/" + str(ws_use) + "/multi_task_pretrain_val_MS_" + varname + ".yaml", "w")
        file_yml_part_var_MS_pre_write_val.write(yml_part_var[varname].replace("features: M", "features: MS").replace("TRAIN", "TRAIN_VAL"))
        file_yml_part_var_MS_pre_write_val.close()
        
        file_yml_part_var_MS_write_val = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_val_MS_" + varname + ".yaml", "w")
        file_yml_part_var_MS_write_val.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_part_var_MS_write_val.close()

        file_yml_part_var_MS_write = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_MS_" + varname + ".yaml", "w")
        file_yml_part_var_MS_write.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_part_var_MS_write.close()
        
        file_yml_part_var_MS_write_train = open("csv_data/data_provider/" + str(ws_use) + "/zeroshot_task_train_MS_" + varname + ".yaml", "w")
        file_yml_part_var_MS_write_train.write(yml_part_var[varname].replace("features: M", "features: MS").replace("pretrain_", ""))
        file_yml_part_var_MS_write_train.close()