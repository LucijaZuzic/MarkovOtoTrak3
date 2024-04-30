from utilities import load_object
import numpy as np

dicti_all = load_object("dicti_all_traj")

rv_metric = {"R2": 2, "RMSE": 6, "MAE": 6, "R2_wt": 2, "RMSE_wt": 6, "MAE_wt": 6, "Euclid": 6}
mul_metric = {"R2": 100, "RMSE": 1, "MAE": 1, "R2_wt": 100, "RMSE_wt": 1, "MAE_wt": 1, "Euclid": 1}
list_ws = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]

for metric_name_use in list(rv_metric.keys()):
    for varname in dicti_all:
        str_pr = ""
        first_line = metric_name_use + " " + varname
        for model_name_use in dicti_all[varname]:
            for val_ws in list_ws:
                first_line += " & $" + str(val_ws) + "$"
            break
        print(first_line + " \\\\ \\hline")
        for model_name_use in dicti_all[varname]:
            str_pr += varname + " " + metric_name_use + " " + model_name_use
            for val_ws in list_ws: 
                vv = dicti_all[varname][model_name_use][str(val_ws)][metric_name_use]  
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)

for metric_name_use in list(rv_metric.keys()):
    for model_name_use in dicti_all["long speed dir"]:
        str_pr = ""
        first_line = metric_name_use + " " + model_name_use
        for varname in dicti_all:
            for val_ws in list_ws:
                first_line += " & $" + str(val_ws) + "$"
            break
        print(first_line + " \\\\ \\hline")
        for varname in dicti_all:
            str_pr += varname + " " + metric_name_use + " " + model_name_use
            for val_ws in list_ws: 
                vv = dicti_all[varname][model_name_use][str(val_ws)][metric_name_use]  
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)