from utilities import load_object
import numpy as np

def get_cat(val):
    if val < 10: return "Almost useless"
    if val < 20: return "Hard to get the gist"
    if val < 30: return "The gist is clear, but has significant grammatical errors"
    if val < 40: return "Understandable to good translations"
    if val < 50: return "High quality translations"
    if val < 60: return "Very high quality, adequate, and fluent translations"
    return "Quality often better than human"

dicti_all = load_object("dicti_all_BLEU")
ord_metric_old = ["GRU_100", "RNN_100", "LSTM_100", "GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4"]
metric_dicti = {"BLEU": 0}
translate_metric = {"BLEU": "BLEU"}
translate_model = {"GRU_100": "GRU model", "RNN_100": "RNN model", "LSTM_100": "LSTM model", "UniTS": "UniTS model", "UniTS_no_time": "UniTS model trained without time intervals", "UniTS_varname": "UniTS model trained separately for each variable",
        "GRU_Att_1": "GRU attention model using the hyperparameters from experiment 1", 
        "GRU_Att_2": "GRU attention model using the hyperparameters from experiment 2", 
        "GRU_Att_3": "GRU attention model using the hyperparameters from experiment 3", 
        "GRU_Att_4": "GRU attention model using the hyperparameters from experiment 4"}
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
translate_ws = {"2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "15": "fifteen", "19": "nineteen", "20": "twenty", "25": "twenty-five", "29": "twenty-nine", "30": "thirty"}
list_ws = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19, 20, 25, 29, 30]
 
#for additional_metric in ["UniTS", "UniTS_no_time", "UniTS_varname"]:
for additional_metric in ["UniTS_no_time"]:
    ord_metric = [x for x in ord_metric_old]
    ord_metric.append(additional_metric)
    for metric_name_use in list(metric_dicti.keys()):
        for varname in dicti_all:
            duplicate_val_all = True
            duplicate_val = True
            mul_metric = 0
            rv_metric = 2
            while duplicate_val_all:
                set_values_all = set()
                set_values = dict()
                for val_ws in list_ws:
                    set_values[val_ws] = set()
                max_col = dict()
                max_col_str = dict()
                max_col_ix = dict()
                for val_ws in list_ws:
                    max_col[val_ws] = -1000000
                    max_col_str[val_ws] = -1
                    max_col_ix[val_ws] = -1
                min_col = dict()
                min_col_str = dict()
                min_col_ix = dict()
                for val_ws in list_ws:
                    min_col[val_ws] = 1000000
                    min_col_str[val_ws] = -1
                    min_col_ix[val_ws] = -1
                duplicate_val_all = False
                duplicate_val = False
                str_pr = ""
                first_line = metric_name_use + " " + varname + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
                for model_name_use in ord_metric:
                    for val_ws in list_ws:
                        first_line += " & $" + str(val_ws) + "$s"
                    break
                for model_name_use in ord_metric:
                    str_pr += model_name_use.replace("_100", "").replace("_", " ")
                    for val_ws in list_ws: 
                        vv = dicti_all[varname][model_name_use][str(val_ws)][metric_name_use]  
                        vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                        str_pr += " & $" + str(vv) + "$"
                        if vv in set_values[val_ws]:
                            duplicate_val = True
                        if vv in set_values_all:
                            duplicate_val_all = True
                        set_values[val_ws].add(vv)
                        set_values_all.add(vv)
                        vv_new = str(vv)
                        if metric_dicti[metric_name_use] == 2:
                            vv_new += "\%"
                        if mul_metric != 0:
                            vv_new += "\\times 10^{-" + str(mul_metric) + "}"
                        if vv > max_col[val_ws]:
                            max_col[val_ws] = vv
                            max_col_str[val_ws] = vv_new
                            max_col_ix[val_ws] = model_name_use
                        if vv < min_col[val_ws]:
                            min_col[val_ws] = vv
                            min_col_str[val_ws] = vv_new
                            min_col_ix[val_ws] = model_name_use
                    str_pr += " \\\\ \\hline\n"
                if duplicate_val_all:
                    rv_metric += 1
                if rv_metric > 3 or mul_metric > 6:
                    break
            max_max = -1000000
            max_max_ix = -1
            max_max_str = -1
            max_max_ws = -1
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
                if max_col[val_ws] > max_max:
                    max_max = max_col[val_ws]
                    max_max_ix = max_col_ix[val_ws]
                    max_max_str = max_col_str[val_ws]
                    max_max_ws = val_ws
                strnew = "The highest " + translate_metric[metric_name_use] + " score of $" + max_col_str[val_ws] + "$ for the " + translate_varname[varname] + " estimated using a window size of " + translate_ws[str(val_ws)] + " seconds was achieved with the " + translate_model[max_col_ix[val_ws]] + ', and falls into the "' + get_cat(max_col[val_ws]) + '" category.'
                #print(strnew)
            strnew = "The highest " + translate_metric[metric_name_use] + " score of $" + max_max_str + "$ for the estimated " + translate_varname[varname] + " was achieved using a window size of " + translate_ws[str(max_max_ws)] + " seconds and the " + translate_model[max_max_ix] + ' and falls into the "' + get_cat(max_max) + '" category.'
            print(strnew)