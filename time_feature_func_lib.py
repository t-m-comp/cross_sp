import os
import numpy as np
from scipy import stats

import math

import setting_value as setV

import ikeda_lib

#############calc trajectory feature fanctions###############

def calc_curves(trajectory_csv):

    x_tjr = trajectory_csv[:,3]
    y_tjr = trajectory_csv[:,4]

    curves_list = [math.pi]
    for i in range(len(x_tjr)-2):

        a = np.array([x_tjr[i], y_tjr[i]])#
        c = np.array([x_tjr[i+1], y_tjr[i+1]])#起点
        b = np.array([x_tjr[i+2], y_tjr[i+2]])#

        ac = a - c
        bc = b - c

        try:
            num = np.dot(ac, bc)
            denom = np.linalg.norm(ac) * np.linalg.norm(bc)
            rad = np.arccos(num/denom)
            #print 'num/denom'+str(num/denom)+'deg:'+str(rad*180.0/math.pi)
            if rad> math.pi:
                rad = 2*math.pi - rad

            if np.isnan(rad):
                rad = 0
        except:
            print(i,a,b,c)
            print("exception in calc_curves")
            #exit()

        curves_list.append(rad)

    curves_list.append(math.pi)

    return np.array(curves_list)

def get_moving_value_one(value_list,middle_time_step,window_size_han,calc_type):

    #limit
    value_max_limit = len(value_list)-1
    value_min_limit = 0

    value_list_buf = []
    for current_time_step in range(middle_time_step-window_size_han,middle_time_step+window_size_han+1):
        if current_time_step<value_min_limit or value_max_limit<current_time_step:
            value_list_buf.append(value_list[middle_time_step])
        else:
            value_list_buf.append(value_list[current_time_step])

    ##by calc_type
    if calc_type=="average":
        return np.mean(value_list_buf)
    elif calc_type=="variance":
        return np.var(value_list_buf)
    elif calc_type=="waido":
        return stats.skew(value_list_buf)
    elif calc_type=="sendo":
        return stats.kurtosis(value_list_buf)
    elif calc_type=="max":
        return np.max(value_list_buf)
    elif calc_type=="min":
        return np.min(value_list_buf)
    else:
        print("ERROR in get_moving_value_one")
        sys.exit()

def calc_moving_values(value_list,window_size_han,calc_type):

    moving_value_list = []
    for middle_time_step in range(len(value_list)):
        moving_value_list.append(get_moving_value_one(value_list,middle_time_step,window_size_han,calc_type))

    return np.array(moving_value_list)

def calc_values(trajectory_csv,value_type,window_size_han,calc_type):

    if value_type=="speed":
        value_list = time_feature_speed(trajectory_csv)
    elif value_type=="angle":
        value_list = time_feature_angle_speed(trajectory_csv)
    elif value_type=="angle_abs":
        value_list = time_feature_angle_speed_abs(trajectory_csv)
    elif value_type=="speed_acc":
        value_list = time_feature_acc(trajectory_csv)
    elif value_type=="angle_acc":
        value_list = time_feature_angle_acc(trajectory_csv)
    elif value_type=="angle_acc_abs":
        value_list = time_feature_angle_acc_abs(trajectory_csv)

    return calc_moving_values(value_list,window_size_han,calc_type)


def time_feature_angle_speed(trajectory_csv):

    return trajectory_csv[:,1]

def time_feature_angle_speed_abs(trajectory_csv):

    return np.abs(trajectory_csv[:,1])

def time_feature_speed(trajectory_csv):

    return trajectory_csv[:,2]

"""def time_feature_x(trajectory_csv):

    return trajectory_csv[:,3]

def time_feature_y(trajectory_csv):

    return trajectory_csv[:,4]"""

def calc_diff(value_list,diff_num):

    value_list_2 = np.append(value_list,value_list[-diff_num:])
    value_list_2 = value_list_2[diff_num:]

    return (value_list_2 - value_list)

def time_feature_acc(trajectory_csv):

    speed_list = time_feature_speed(trajectory_csv)

    speed_list_2 = np.append(speed_list,speed_list[-1])
    speed_list_2 = np.delete(speed_list_2,0)

    return (speed_list_2 - speed_list)

def time_feature_moving_average_speed(trajectory_csv):

    speed_list = time_feature_speed(trajectory_csv)

    return calc_moving_values(speed_list,10,"average")

def time_feature_moving_variance_speed(trajectory_csv):

    speed_list = time_feature_speed(trajectory_csv)

    return calc_moving_values(speed_list,10,"variance")

def time_feature_moving_average_angle_abs(trajectory_csv):

    angle_list = time_feature_angle_speed_abs(trajectory_csv)

    return calc_moving_values(angle_list,10,"average")

def time_feature_moving_variance_angle_abs(trajectory_csv):

    angle_list = time_feature_angle_speed_abs(trajectory_csv)

    return calc_moving_values(angle_list,5,"variance")

def time_feature_moving_average_angle(trajectory_csv):

    angle_list = time_feature_angle_speed(trajectory_csv)

    return calc_moving_values(angle_list,10,"average")

def time_feature_moving_variance_angle(trajectory_csv):

    angle_list = time_feature_angle_speed(trajectory_csv)

    return calc_moving_values(angle_list,10,"variance")

def time_feature_angle_acc(trajectory_csv):

    angle_list = time_feature_angle_speed(trajectory_csv)

    angle_list_2 = np.append(angle_list,angle_list[-1])
    angle_list_2 = np.delete(angle_list_2,0)

    return (angle_list_2 - angle_list)

def time_feature_angle_acc_abs(trajectory_csv):

    return np.abs(time_feature_angle_acc(trajectory_csv))
