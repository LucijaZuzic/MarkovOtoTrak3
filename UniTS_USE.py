from utilities import load_object
import os
import numpy as np
from datetime import timedelta, datetime
import pandas as pd

ws_range = range(4, 6)

for ws_use in ws_range:

    if not os.path.isdir("csv_data/data_provider/" + str(ws_use)):
        os.makedirs("csv_data/data_provider/" + str(ws_use))
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        if not os.path.isdir("csv_data/dataset/" + str(ws_use) + "/" + varname):
            os.makedirs("csv_data/dataset/" + str(ws_use) + "/" + varname)

        if not os.path.isdir("tmp_data/"):
            os.makedirs("tmp_data/")

        dictio = {"task_name": "pretrain_long_term_forecast", 
                  "dataset": varname, 
                  "data": "custom", 
                  "embed": "timeF", 
                  "root_path": "tmp_data/", 
                  "data_path": "use_data.csv", 
                  "features": "M",
                  "seq_len": ws_use, 
                  "label_len": 0, 
                  "pred_len": 1, 
                  "enc_in": ws_use, 
                  "dec_in": ws_use, 
                  "c_out": ws_use}
         
        yml_part = "task_dataset:"
        yml_part += "\n " + str(varname) + ":" 
        for v in dictio:
            yml_part += "\n  " + v + ": " + str(dictio[v]) 
        yml_part += "\n"

        file_pd = pd.read_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "newdata_TEST.csv", index_col = False)
        new_dict = dict()
        for new_ix in range(len(file_pd["date"])):
            new_dict["date"] = file_pd["date"][new_ix]
            for num in range(ws_use):
                new_dict[str(num)] = file_pd[str(num)][new_ix]
            new_dict["OT"] = file_pd["OT"][new_ix]
            break
        
        line = "python run.py --is_training 0"
        line += "--model_id UniTS_zeroshot_pretrain_x64_mine_all_new_" + str(ws_use) + "_test "
        line += "--model UniTS_zeroshot --prompt_num 10 --patch_len 1 --stride 1"
        line += "--e_layers 3 --d_model 64 --des 'Exp' --debug online"
        line += "--project_name zeroshot_newdata_mine_all_new_" + str(ws_use - 1) + "_test"
        line += "--pretrained_weight checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_" + str(ws_use) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth  "
        line += "--task_data_config_path data_provider/4/zeroshot_task.yaml"


