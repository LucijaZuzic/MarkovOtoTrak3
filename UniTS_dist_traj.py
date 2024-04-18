import os
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_object, save_object, compare_traj_and_sample
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ws_range = range(5, 7)

def str_convert_new(val):
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    rounded = "$" + str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded + "$"

def new_metric_translate(metric_name):
    new_metric_name = {"trapz x": "$x$ integration", 
              "trapz y": "$y$ integration",
              "euclidean": "Euclidean distance"}
    if metric_name in new_metric_name:
        return new_metric_name[metric_name]
    else:
        return metric_name
    
def translate_category(long):
    translate_name = {
        "long no abs": "$x$ and $y$ offset",  
        "long speed dir": "Speed, heading, and time", 
        "long speed ones dir": "Speed, heading, and a 1s time interval", 
    }
    if long in translate_name:
        return translate_name[long]
    else:
        return long
    
def draw_mosaic(rides_actual, rides_predicted, name):
    
    x_dim_rides = int(np.sqrt(len(rides_actual)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides_actual):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)

    for ix_ride in range(len(rides_actual)):
 
        x_actual, y_actual = rides_actual[ix_ride]["long"], rides_actual[ix_ride]["lat"]
        x_predicted, y_predicted = rides_predicted[ix_ride]["long"], rides_predicted[ix_ride]["lat"]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.rcParams.update({'font.size': 28}) 
        plt.rcParams['font.family'] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.axis("equal")
        plt.axis("off")

        plt.plot(x_predicted, y_predicted, c = "b", linewidth = 2, label = "Estimated")
    
        plt.plot(x_actual, y_actual, c = "k", linewidth = 2, label = "Original")

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def draw_mosaic_one(x_actual, y_actual, x_predicted, y_predicted, k, model_name, name, ws_use, dist_name):
     
    plt.figure(figsize = (10, 10), dpi = 80)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")
  
    plt.plot(x_predicted, y_predicted, c = "b", linewidth = 10, label = "Estimated")
  
    plt.plot(x_actual, y_actual, c = "k", linewidth = 10, label = "Original")

    plt.plot(x_actual[0], x_actual[0], marker = "o", label = "Start", color = "k", mec = "k", mfc = "g", ms = 20, mew = 10, linewidth = 10) 
   
    split_file_veh = k.split("/")
    vehicle = split_file_veh[0].replace("Vehicle_", "")
    ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")
  
    title_new = "Vehicle " + vehicle + " Ride " + ride + "\n" + model_name + " model\n" + "Window size " + str(ws_use) + "\n"

    title_new += translate_category(dist_name) + "\n" 
    for metric in distance_predicted_new:
        if "simpson" in metric:
            continue
        title_new += new_metric_translate(metric) + ": " + str_convert_new(distance_predicted_new[metric][model_name][ws_use][dist_name][k]) + "\n"

    plt.title(title_new)
    plt.legend()
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
predicted_all = load_object("UniTS_final_result/predicted_all")
y_test_all = load_object("UniTS_final_result/y_test_all")
ws_all = load_object("UniTS_final_result/ws_all")

actual_long = load_object("UniTS_final_result/actual_long")
actual_lat = load_object("UniTS_final_result/actual_lat")
predicted_long = load_object("UniTS_final_result/predicted_long")
predicted_lat = load_object("UniTS_final_result/predicted_lat")

distance_predicted_new = dict()

metric_names = ["euclidean"] 

if not os.path.isdir("mosaic_UniTS"):
    os.makedirs("mosaic_UniTS")

if not os.path.isdir("mosaic_UniTS_all"):
    os.makedirs("mosaic_UniTS_all")

use_draw = True

for metric in metric_names:

    distance_predicted_new[metric] = dict()

    for model_name in predicted_long:

        distance_predicted_new[metric][model_name] = dict()

        for ws_use in predicted_long[model_name]:

            distance_predicted_new[metric][model_name][ws_use] = dict()
    
            for dist_name in predicted_long[model_name][ws_use]:

                distance_predicted_new[metric][model_name][ws_use][dist_name] = dict()
                
                all_actual = []
                all_predicted = []

                actual_long_lat = []
                actual_long_lat_time = []
                predicted_long_lat = []
                predicted_long_lat_time = []

                vals_avg = []
                
                int_veh = sorted([int(k.split("/")[0].split("_")[1]) for k in predicted_long[model_name][ws_use][dist_name].keys()])

                for v in set(int_veh):

                    all_actual_vehicle = []
                    all_predicted_vehicle = []
                    
                    for k in predicted_long[model_name][ws_use][dist_name]:
                        veh_new = int(k.split("/")[0].split("_")[1])
                        
                        if veh_new != v:
                            continue 

                        actual_long_one = actual_long[model_name][ws_use][k]
                        actual_lat_one = actual_lat[model_name][ws_use][k]

                        predicted_long_one = predicted_long[model_name][ws_use][dist_name][k]
                        predicted_lat_one = predicted_lat[model_name][ws_use][dist_name.replace("long", "lat")][k]

                        use_len = min(len(actual_long_one), len(predicted_long_one))
                        
                        actual_long_one = actual_long_one[:use_len]
                        actual_lat_one = actual_lat_one[:use_len]

                        predicted_long_one = predicted_long_one[:use_len]
                        predicted_lat_one = predicted_lat_one[:use_len]
                        
                        time_actual = y_test_all["time"][model_name][ws_use][k]
                        time_predicted = predicted_all["time"][model_name][ws_use][k]

                        time_actual_cumulative = [0]
                        time_predicted_cumulative = [0]
                        
                        for ix in range(len(time_actual)):
                            time_actual_cumulative.append(time_actual_cumulative[-1] + time_actual[ix])
                            time_predicted_cumulative.append(time_predicted_cumulative[-1] + time_predicted[ix])
                            
                        use_len_time = min(use_len, len(time_actual_cumulative))

                        actual_long_one = actual_long_one[:use_len_time]
                        actual_lat_one = actual_lat_one[:use_len_time]

                        predicted_long_one = predicted_long_one[:use_len_time]
                        predicted_lat_one = predicted_lat_one[:use_len_time]
                        
                        time_actual_cumulative = time_actual_cumulative[:use_len_time]
                        time_predicted_cumulative = time_predicted_cumulative[:use_len_time]

                        distance_predicted_new[metric][model_name][ws_use][dist_name][k] = compare_traj_and_sample(actual_long_one, actual_lat_one, time_actual_cumulative, {"long": predicted_long_one, "lat": predicted_lat_one, "time": time_predicted_cumulative}, metric)
                        
                        split_file_veh = k.split("/")
                        vehicle = split_file_veh[0].replace("Vehicle_", "")
                        ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

                        if use_draw:

                            filename = "mosaic_UniTS_all/Vehicle_" + vehicle + "_events_" + ride + "_" + model_name + "_" + str(ws_use) + "_" + dist_name + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                            draw_mosaic_one(actual_long_one, actual_lat_one, predicted_long_one, predicted_lat_one, k, model_name, filename, ws_use, dist_name)

                        all_actual.append({"long": actual_long_one, "lat": actual_lat_one})
                        all_predicted.append({"long": predicted_long_one, "lat": predicted_lat_one})

                        vals_avg.append(distance_predicted_new[metric][model_name][ws_use][dist_name][k])

                        all_actual_vehicle.append({"long": actual_long_one, "lat": actual_lat_one})
                        all_predicted_vehicle.append({"long": predicted_long_one, "lat": predicted_lat_one})

                        for ix_use_len in range(use_len_time):

                            actual_long_lat.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len]])
                            actual_long_lat_time.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])
                            
                            predicted_long_lat.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len]])
                            predicted_long_lat_time.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

                    if use_draw:

                        filename_veh = "mosaic_UniTS/Vehicle_" + str(v) + "_" + model_name + "_" + str(ws_use) + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                        draw_mosaic(all_actual_vehicle, all_predicted, filename_veh)

                if use_draw:

                    filename = "mosaic_UniTS/all_" + model_name + "_" + str(ws_use) + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                    draw_mosaic(all_actual, all_predicted, filename)

                print(model_name + "_" + str(ws_use) + "_" + dist_name, np.round(np.average(vals_avg), 6))

                r2_pred_wt = r2_score(actual_long_lat_time, predicted_long_lat_time)

                mae_pred_wt = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)

                rmse_pred_wt = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))

                r2_pred = r2_score(actual_long_lat, predicted_long_lat)
        
                mae_pred = mean_absolute_error(actual_long_lat, predicted_long_lat)

                rmse_pred = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
                
                print("R2", np.round(r2_pred * 100, 2))
                print("MAE", np.round(mae_pred, 6))
                print("RMSE", np.round(rmse_pred, 6))

                print("R2_wt", np.round(r2_pred_wt * 100, 2))
                print("MAE_wt", np.round(mae_pred_wt, 6))
                print("RMSE_wt", np.round(rmse_pred_wt, 6))

save_object("UniTS_final_result/distance_predicted_new", distance_predicted_new)