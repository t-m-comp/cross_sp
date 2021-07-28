import numpy as np
import math
import os
import ikeda_lib

#####normalize_class#####

class not_normalization:

    def __init__(self,params_list,valued_feature_func_name):
        [self.layer_node,self.domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag] = params_list
        self.valued_feature_func_name = valued_feature_func_name

    def normalize_value(self,value):
        return value##sonomama return

class normal_normalization:

    def __init__(self,params_list,valued_feature_func_name):
        [self.layer_node,self.domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag] = params_list
        self.valued_feature_func_name = valued_feature_func_name
        #get valued_list
        valued_feature_list_all = []
        domain_class_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node)
        for domain_class in domain_class_list:
            file_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class)
            for one_filename in file_list:

                #csv yomikomi
                #activated_trajectory
                activated_trajectory_data = ikeda_lib.read_csv(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class+"/"+one_filename)
                activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

                #trajectory
                trajectory_data = ikeda_lib.read_csv(self.trajectory_dir+"/"+domain_class+"/"+one_filename)
                trajectory_data = trajectory_data[1:].astype("float32")

                #trajectory_feature
                trajectory_feature_data = ikeda_lib.read_csv(self.trajectory_feature_dir+"/"+domain_class+"/"+one_filename)
                trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

                ##make set
                trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,one_filename]

                valued_feature_list = np.array([])
                current_params_list = [[self.layer_node,domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag]]
                plist = params_list
                tdatas = trajectory_data_set
                valued_feature_list = eval(valued_feature_func_name)
                valued_feature_list_all.append(valued_feature_list)

        #calc normalize param
        valued_feature_list_all = np.array(valued_feature_list_all)
        self.value_max = np.max(valued_feature_list_all)
        self.value_min = np.min(valued_feature_list_all)

    def normalize_value(self,value):
        return ((value-self.value_min)/(self.value_max-self.value_min))#0~1 ni osamaru

class Dbetsu_normal_normalization:

    def __init__(self,params_list,valued_feature_func_name):
        [self.layer_node,self.domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag] = params_list
        self.valued_feature_func_name = valued_feature_func_name
        #get valued_list
        valued_feature_list_all = []
        domain_class_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node)
        for domain_class in domain_class_list:
            if domain_class!=self.domain_class:
                continue
            file_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class)
            for one_filename in file_list:

                #csv yomikomi
                #activated_trajectory
                activated_trajectory_data = ikeda_lib.read_csv(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class+"/"+one_filename)
                activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

                #trajectory
                trajectory_data = ikeda_lib.read_csv(self.trajectory_dir+"/"+domain_class+"/"+one_filename)
                trajectory_data = trajectory_data[1:].astype("float32")

                #trajectory_feature
                trajectory_feature_data = ikeda_lib.read_csv(self.trajectory_feature_dir+"/"+domain_class+"/"+one_filename)
                trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

                ##make set
                trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,one_filename]

                valued_feature_list = np.array([])
                current_params_list = [[self.layer_node,domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag]]
                plist = params_list
                tdatas = trajectory_data_set
                valued_feature_list = eval(valued_feature_func_name)
                valued_feature_list_all.append(valued_feature_list)

        #calc normalize param
        valued_feature_list_all = np.array(valued_feature_list_all)
        self.value_max = np.max(valued_feature_list_all)
        self.value_min = np.min(valued_feature_list_all)

    def normalize_value(self,value):
        return ((value-self.value_min)/(self.value_max-self.value_min))#0~1 ni osamaru

    """#calc normalize param
        valued_feature_list_all = np.array(valued_feature_list_all)
        self.value_mean = np.mean(valued_feature_list_all)
        self.value_std = np.std(valued_feature_list_all)

    def normalize_value(self,value):
        return ((value-self.value_mean)/(3*self.value_std))+0.5#oyoso 0~1 ni osamaru hazu"""


class zerowari_normalization:

    def __init__(self,params_list,valued_feature_func_name):
        [self.layer_node,self.domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag] = params_list
        self.params_list = params_list
        self.valued_feature_func_name = valued_feature_func_name
        #get valued_list
        valued_feature_list_all = []
        domain_class_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node)
        for domain_class in domain_class_list:
            file_list = os.listdir(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class)
            for one_filename in file_list:

                #csv yomikomi
                #activated_trajectory
                activated_trajectory_data = ikeda_lib.read_csv(self.activated_trajectory_dir+"/"+self.layer_node+"/"+domain_class+"/"+one_filename)
                activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

                #trajectory
                trajectory_data = ikeda_lib.read_csv(self.trajectory_dir+"/"+domain_class+"/"+one_filename)
                trajectory_data = trajectory_data[1:].astype("float32")

                #trajectory_feature
                trajectory_feature_data = ikeda_lib.read_csv(self.trajectory_feature_dir+"/"+domain_class+"/"+one_filename)
                trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

                ##make set
                trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,one_filename]

                valued_feature_list = np.array([])
                current_params_list = [[self.layer_node,domain_class],[self.trajectory_dir,self.trajectory_feature_dir,self.activated_trajectory_dir,self.output_image_dir],[self.model_dirname,self.use_unit_flag]]
                plist = params_list
                tdatas = trajectory_data_set
                valued_feature_list = eval(valued_feature_func_name)
                valued_feature_list_all.append(valued_feature_list)

        #calc normalize param
        valued_feature_list_all = np.array(valued_feature_list_all)
        #get_plus_only:|2|-5|=>|2|0|
        plus_only_valued_feature_list_all = (valued_feature_list_all+np.abs(valued_feature_list_all))/2
        self.plus_value_max = np.max(plus_only_valued_feature_list_all)
        #get_minus_only:|2|-5|=>|0|-5|
        minus_only_valued_feature_list_all = (valued_feature_list_all-np.abs(valued_feature_list_all))/2
        self.minus_value_min = np.min(minus_only_valued_feature_list_all)

    def normalize_value(self,value):
        if value>=0:
            value = value/float(self.plus_value_max)#0~1
        else:
            value = value/float(np.abs(self.minus_value_min))#-1~0

        return (value+1)/float(2)#0~1

#####valued_features#####
##utility##
def replace_value_func(value_list,replace_list):

    replace_list_one_memori = float(1)/len(replace_list)
    value_list_replaced = []
    for i in range(len(value_list)):
        replace_list_num = int( (value_list[i]+1)/(float(2))/replace_list_one_memori)

        if replace_list_num==len(replace_list):
            replace_list_num=len(replace_list)-1
        value_list_replaced.append(replace_list[replace_list_num])

    return np.array(value_list_replaced)

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
    else:
        print("ERROR in get_moving_value_one")
        sys.exit()

def calc_moving_values(value_list,window_size_han,calc_type):

    moving_value_list = []
    for middle_time_step in range(len(value_list)):
        moving_value_list.append(get_moving_value_one(value_list,middle_time_step,window_size_han,calc_type))

    return np.array(moving_value_list)
##

def normal_activation(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    activation_list = activated_trajectory_data[:,5]

    return activation_list

def get_value_params(params_list,trajectory_data_set,value_number):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    return activated_trajectory_data[:,value_number]

def normal_attention1(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    attention_list = activated_trajectory_data[:,5]

    return attention_list

def normal_attention2(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    attention_list = activated_trajectory_data[:,6]

    return attention_list

def normal_state(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    activation_list = activated_trajectory_data[:,6]

    return activation_list

def hist_diff_activation(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    activation_list = activated_trajectory_data[:,5]

    activation_dist_csv = ikeda_lib.read_csv(model_dirname+"/activation_dist/csv/"+layer_node+"/all_domain.csv").astype("float32")
    activation_dist_diff = activation_dist_csv[0]-activation_dist_csv[1]##normal_yuui:+/dop_yuui:-

    return replace_value_func(activation_list,activation_dist_diff)

def hist_diff_moving_activation(params_list,trajectory_data_set):

    ##settings##
    window_size = 30
    hist_diff_activation_list = hist_diff_activation(params_list,trajectory_data_set)

    return calc_moving_values(hist_diff_activation_list,window_size,"average")

def hist_diff_moving_activation_sabun(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set

    ##get main_node
    main_value_list = hist_diff_moving_activation(params_list,trajectory_data_set)

    ##settings##
    minus_node = "layer1-node19"
    #get minus_node

    #csv yomikomi
    #activated_trajectory
    activated_trajectory_data = ikeda_lib.read_csv(activated_trajectory_dir+"/"+minus_node+"/"+domain_class+"/"+filename)
    activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

    #trajectory
    trajectory_data = ikeda_lib.read_csv(trajectory_dir+"/"+domain_class+"/"+filename)
    trajectory_data = trajectory_data[1:].astype("float32")

    #trajectory_feature
    trajectory_feature_data = ikeda_lib.read_csv(trajectory_feature_dir+"/"+domain_class+"/"+filename)
    trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

    #make set
    trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,filename]
    minus_params_list = [[minus_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname]]

    minus_value_list = hist_diff_moving_activation(minus_params_list,trajectory_data_set)

    return ( np.where(main_value_list>=3000,10000,0)+np.where(main_value_list<=-2000,-10000,0) ) - ( np.where(minus_value_list>=40,10000,0)+np.where(minus_value_list<=-300,-10000,0) )#( np.where(main_value_list>=4000,10000,0)+np.where(main_value_list<=-3000,-10000,0) ) - ( np.where(minus_value_list>=550,10000,0)+np.where(minus_value_list<=-400,-10000,0) )##

"""def hist_diff_moving_activation_threshold(params_list,trajectory_data_set):

    hist_diff_activation_list = hist_diff_activation(params_list,trajectory_data_set)

    return np.where(hist_diff_activation_list<-0.1,100,0)+np.where(hist_diff_activation_list>-0.1,-100,0)"""

def bandpass_activation(params_list,trajectory_data_set):

    normal_activation_list = normal_activation(params_list,trajectory_data_set)

    #activation_highlight_list = np.where(normal_activation_list>=0.73,1,0)*np.where(normal_activation_list<=0.85,1,0)+np.where(normal_activation_list>=-0.15,1,0)*np.where(normal_activation_list<=0.73,1,0)*2
    activation_highlight_list = np.where(normal_activation_list>=-0.25,1,0)*np.where(normal_activation_list<=0.25,1,0)

    return activation_highlight_list

def bandpass_activation2(params_list,trajectory_data_set):

    normal_activation_list = normal_activation(params_list,trajectory_data_set)
    speed_list = moving_avr_speed_normalized(params_list,trajectory_data_set)

    #speed_highlight_list = np.where(speed_list<=-0.25,1,0)+np.where(speed_list>=0.75,1,0)

    activation_highlight_list = np.where(normal_activation_list<=-0.7,1,0)*np.where(normal_activation_list>=-0.9,1,0)*1+np.where(normal_activation_list>=-0.15,1,0)*np.where(normal_activation_list<=1.0,1,0)*2

    return activation_highlight_list#*speed_highlight_list

def bandpass_activation3(params_list,trajectory_data_set):

    normal_activation_list = normal_activation(params_list,trajectory_data_set)
    speed_list = moving_avr_speed_normalized(params_list,trajectory_data_set)

    activation_highlight_list = np.where(normal_activation_list<=-0.7,1,0)*np.where(normal_activation_list>=-0.9,1,0)*1+np.where(normal_activation_list>=-0.15,1,0)*np.where(normal_activation_list<=1.0,1,0)*2

    speed_highlight_list = np.where(speed_list>-0.25,1,0)*np.where(speed_list<0.75,1,0)

    highlight_list = activation_highlight_list*speed_highlight_list

    return highlight_list

#####graph_features#####
def moving_avr_speed_normalized(params_list,trajectory_data_set):

    ##settings##
    window_size = 5
    speed_normalized_list = speed_normalized(params_list,trajectory_data_set)

    return calc_moving_values(speed_normalized_list,window_size,"average")

def moving_avr_angle_speed_normalized(params_list,trajectory_data_set):

    ##settings##
    window_size = 2
    angle_speed_normalized_list = np.abs(angle_speed_normalized(params_list,trajectory_data_set))

    return calc_moving_values(angle_speed_normalized_list,window_size,"average")

def moving_avr_angle_speed_normalized2(params_list,trajectory_data_set):

    ##settings##
    window_size = 25
    angle_speed_normalized_list = angle_speed_normalized(params_list,trajectory_data_set)

    return calc_moving_values(angle_speed_normalized_list,window_size,"average")

def moving_avr_angle_speed_normalized3(params_list,trajectory_data_set):

    ##settings##
    window_size = 50
    angle_speed_normalized_list = angle_speed_normalized(params_list,trajectory_data_set)

    return calc_moving_values(angle_speed_normalized_list,window_size,"average")

def moving_var_angle_speed_normalized(params_list,trajectory_data_set):

    ##settings##
    window_size = 5
    angle_speed_normalized_list = np.abs(angle_speed_normalized(params_list,trajectory_data_set))

    return calc_moving_values(angle_speed_normalized_list,window_size,"variance")

def speed_normalized(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    speed_normalized_list = trajectory_feature_data[:,2]

    return speed_normalized_list

def acc_normalized(params_list,trajectory_data_set):

    speed_normalized_list = speed_normalized(params_list,trajectory_data_set)

    return speed_normalized_list-np.hstack((speed_normalized_list[1:],np.array([speed_normalized_list[-1]])))

def angle_speed_normalized(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    angle_speed_normalized_list = trajectory_feature_data[:,1]

    return angle_speed_normalized_list

def activation_feature(params_list,trajectory_data_set):

    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list
    activated_trajectory_data,trajectory_data,trajectory_feature_data,filename = trajectory_data_set
    ###

    activation_feature_list = activated_trajectory_data[:,5]

    return activation_feature_list
