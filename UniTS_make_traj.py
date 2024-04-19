import pandas as pd
import os  
from utilities import load_object, save_object, get_sides_from_angle
from pytorch_utilities import get_XY

def change_angle(angle, name_file):
    
    file_with_ride = pd.read_csv(name_file) 
    
    x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
    y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

predicted_all = dict()
y_test_all = dict()
ws_all = dict() 

ws_range = range(4, 7)
 
model_name = "UniTS"

for varname in os.listdir("final_train_pytorch"):

    predicted_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 

    predicted_all[varname][model_name] = dict()
    y_test_all[varname][model_name] = dict()
    ws_all[varname][model_name] = dict() 
    
    for ws_use in ws_range:

        predicted_all[varname][model_name][ws_use] = dict()
        y_test_all[varname][model_name][ws_use] = dict()
        ws_all[varname][model_name][ws_use] = dict() 
  
        final_test_data = pd.read_csv("UniTS_final_res/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

        file_object_test = load_object("actual/actual_" + varname)

        len_total = 0

        for k in file_object_test:
            
            ws_all[varname][model_name][ws_use][k] = ws_use
            
            x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, 1)
            
            y_test_all[varname][model_name][ws_use][k] = []
            for ix1 in range(len(y_test_part)): 
                for ix2 in range(len(y_test_part[ix1])): 
                    y_test_all[varname][model_name][ws_use][k].append(y_test_part[ix1][ix2])

            predicted_all[varname][model_name][ws_use][k] = list(final_test_data["predicted"][len_total:len_total + len(y_test_all[varname][model_name][ws_use][k])])
            len_total += len(y_test_all[varname][model_name][ws_use][k])  

        if not os.path.isdir("UniTS_final_result"):
            os.makedirs("UniTS_final_result")

        save_object("UniTS_final_result/predicted_all", predicted_all)
        save_object("UniTS_final_result/y_test_all", y_test_all)
        save_object("UniTS_final_result/ws_all", ws_all)

predicted_long = dict()
predicted_lat = dict()

actual_long = dict()
actual_lat = dict() 
        
actual_long[model_name] = dict()
actual_lat[model_name] = dict()

predicted_long[model_name] = dict()
predicted_lat[model_name] = dict()

for ws_use in ws_range:

    actual_long[model_name][ws_use] = dict()
    actual_lat[model_name][ws_use] = dict()

    predicted_long[model_name][ws_use] = dict()
    predicted_lat[model_name][ws_use] = dict()
   
    for k in y_test_all["longitude_no_abs"][model_name][ws_use]:
        print(model_name, k, "actual")
        actual_long[model_name][ws_use][k] = [0]
        actual_lat[model_name][ws_use][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][ws_use][k], ws_all["latitude_no_abs"][model_name][ws_use][k])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][ws_use][k]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][ws_use][k]
        range_long = len(y_test_all["longitude_no_abs"][model_name][ws_use][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][ws_use][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            actual_long[model_name][ws_use][k].append(actual_long[model_name][ws_use][k][-1] + y_test_all["longitude_no_abs"][model_name][ws_use][k][ix + long_offset])
            actual_lat[model_name][ws_use][k].append(actual_lat[model_name][ws_use][k][-1] + y_test_all["latitude_no_abs"][model_name][ws_use][k][ix + lat_offset])

    predicted_long[model_name][ws_use] = dict()
    predicted_lat[model_name][ws_use] = dict()
        
    predicted_long[model_name][ws_use]["long no abs"] = dict()
    predicted_lat[model_name][ws_use]["lat no abs"] = dict()

    for k in predicted_all["longitude_no_abs"][model_name][ws_use]:
        print(model_name, k, "long no abs")
        predicted_long[model_name][ws_use]["long no abs"][k] = [0]
        predicted_lat[model_name][ws_use]["lat no abs"][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][ws_use][k], ws_all["latitude_no_abs"][model_name][ws_use][k])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][ws_use][k]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][ws_use][k]
        range_long = len(y_test_all["longitude_no_abs"][model_name][ws_use][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][ws_use][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            predicted_long[model_name][ws_use]["long no abs"][k].append(predicted_long[model_name][ws_use]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][ws_use][k][ix + long_offset])
            predicted_lat[model_name][ws_use]["lat no abs"][k].append(predicted_lat[model_name][ws_use]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][ws_use][k][ix + lat_offset])

    predicted_long[model_name][ws_use]["long speed dir"] = dict()
    predicted_lat[model_name][ws_use]["lat speed dir"] = dict()

    for k in predicted_all["speed"][model_name][ws_use]:
        print(model_name, k, "long speed dir")
        predicted_long[model_name][ws_use]["long speed dir"][k] = [0]
        predicted_lat[model_name][ws_use]["lat speed dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][ws_use][k], ws_all["direction"][model_name][ws_use][k]), ws_all["time"][model_name][ws_use][k])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][ws_use][k]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][ws_use][k]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][ws_use][k]
        range_speed_time = len(y_test_all["speed"][model_name][ws_use][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][ws_use][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][ws_use][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][ws_use][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][ws_use][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][ws_use][k][ix + dir_offset_time], k))
            predicted_long[model_name][ws_use]["long speed dir"][k].append(predicted_long[model_name][ws_use]["long speed dir"][k][-1] + new_long)
            predicted_lat[model_name][ws_use]["lat speed dir"][k].append(predicted_lat[model_name][ws_use]["lat speed dir"][k][-1] + new_lat)
            
    predicted_long[model_name][ws_use]["long speed ones dir"] = dict()
    predicted_lat[model_name][ws_use]["lat speed ones dir"] = dict()

    for k in predicted_all["speed"][model_name][ws_use]:
        print(model_name, k, "long speed ones dir")
        predicted_long[model_name][ws_use]["long speed ones dir"][k] = [0]
        predicted_lat[model_name][ws_use]["lat speed ones dir"][k] = [0]
    
        max_offset_speed_dir = max(ws_all["speed"][model_name][ws_use][k], ws_all["direction"][model_name][ws_use][k])
        speed_offset = max_offset_speed_dir - ws_all["speed"][model_name][ws_use][k]
        dir_offset = max_offset_speed_dir - ws_all["direction"][model_name][ws_use][k]
        range_speed = len(y_test_all["speed"][model_name][ws_use][k]) - speed_offset
        range_dir = len(y_test_all["direction"][model_name][ws_use][k]) - dir_offset
        min_range_speed_dir = min(range_speed, range_dir)

        for ix in range(min_range_speed_dir):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][ws_use][k][ix + speed_offset] / 111 / 0.1 / 3600, change_angle(predicted_all["direction"][model_name][ws_use][k][ix + dir_offset], k))
            predicted_long[model_name][ws_use]["long speed ones dir"][k].append(predicted_long[model_name][ws_use]["long speed ones dir"][k][-1] + new_long)
            predicted_lat[model_name][ws_use]["lat speed ones dir"][k].append(predicted_lat[model_name][ws_use]["lat speed ones dir"][k][-1] + new_lat)

    if not os.path.isdir("UniTS_final_result"):
        os.makedirs("UniTS_final_result")

    save_object("UniTS_final_result/actual_long", actual_long)
    save_object("UniTS_final_result/actual_lat", actual_lat)
    save_object("UniTS_final_result/predicted_long", predicted_long)
    save_object("UniTS_final_result/predicted_lat", predicted_lat)