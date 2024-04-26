from utilities import load_object
import os
import numpy as np
import pandas as pd
import math

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
ws_range = [9, 19, 29]

for ws_use in ws_range:

    if not os.path.isdir("csv_data/data_provider/" + str(ws_use)):
        os.makedirs("csv_data/data_provider/" + str(ws_use))
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")
 
        file_pd = pd.read_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TEST.csv", index_col= False)
        OT_vals = [list(file_pd["OT"][ix_sth-4:ix_sth+1]) for ix_sth in range(4, len(file_pd["OT"]), 5)]
         
        file_pd_pred = np.array(load_object("../UniTS/results/all_" + str(ws_use) + "_test/preds_" + varname)).reshape(-1)
        file_pd_true = np.array(load_object("../UniTS/results/all_" + str(ws_use) + "_test/trues_" + varname)).reshape(-1)

        file_pd_transformed_pred = np.array(load_object("../UniTS/results/all_" + str(ws_use) + "_test/preds_transformed_" + varname)).reshape(-1)
        file_pd_transformed_true = np.array(load_object("../UniTS/results/all_" + str(ws_use) + "_test/trues_transformed_" + varname)).reshape(-1)
        continue
        dictio = dict()
        for ix_use in range(len(file_pd_transformed_true)):
            val = file_pd_transformed_true[ix_use]
            if val not in dictio:
                dictio[val] = []
            dictio[val].append(file_pd_transformed_pred[ix_use])

        keys_arr = sorted(dictio.keys())
        dictio_close = dict()
        for smv in file_pd["OT"]:
            if smv not in dictio_close: 
                if smv in dictio:
                    vu = smv
                else:
                    vu = find_nearest(keys_arr, smv)
                dictio_close[smv] = vu
 
        preds_smv = []
        actual_smv = [] 
        for smv in file_pd["OT"]:
            preds_smv.append(np.average(dictio[dictio_close[smv]]))
            actual_smv.append(dictio_close[smv])

        if not os.path.isdir("UniTS_final_res/" + str(ws_use)):
            os.makedirs("UniTS_final_res/" + str(ws_use))
  
        df_new = pd.DataFrame({"predicted": preds_smv, "actual": actual_smv})

        df_new.to_csv("UniTS_final_res/" + str(ws_use) + "/" + varname + ".csv", index = False) 

        print(file_pd.columns)