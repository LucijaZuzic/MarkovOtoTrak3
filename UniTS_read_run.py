from utilities import load_object
import os
import numpy as np
from datetime import timedelta, datetime
import pandas as pd

ws_range = range(5, 7)

for ws_use in ws_range:

    if not os.path.isdir("csv_data/data_provider/" + str(ws_use)):
        os.makedirs("csv_data/data_provider/" + str(ws_use))
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")
 
        file_pd = pd.read_csv("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TEST.csv", index_col= False)
        OT_vals = [list(file_pd["OT"][ix_sth-4:ix_sth+1]) for ix_sth in range(4, len(file_pd["OT"]), 5)]
         
        file_pd_pred = np.array(load_object("results/all_" + str(ws_use) + "_test/preds_" + varname)).reshape(-1)
        file_pd_true = np.array(load_object("results/all_" + str(ws_use) + "_test/trues_" + varname)).reshape(-1)

        file_pd_transformed_pred = np.array(load_object("results/all_" + str(ws_use) + "_test/preds_transformed_" + varname)).reshape(-1)
        file_pd_transformed_true = np.array(load_object("results/all_" + str(ws_use) + "_test/trues_transformed_" + varname)).reshape(-1)

        dictio = dict()
        for ix_use in range(len(file_pd_transformed_true)):
            val = file_pd_transformed_true[ix_use]
            if val not in dictio:
                dictio[val] = []
            dictio[val].append(file_pd_transformed_pred[ix_use])
