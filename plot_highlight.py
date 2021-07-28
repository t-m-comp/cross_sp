import sys
import csv
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as clr
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy.matlib as matlib
import os
import io
import gc

from multiprocessing import Pool
from memory_profiler import profile
import shutil

import setting_value as setV
import ikeda_lib
from easy_hist_lib import *

from time_feature_func_lib import *
#from fft_analysis import FFT_make_function
from common_feature_func_lib import *

import inspect

#color_bar_resize = 11#14?#paper用は3

output_number = 40

def spTub(str):
    #space to underbar#
    return "_".join(str.split(" "))

def tr_display_name(function_name,domain_class,use_unit_flag):

    function_params = function_name.split("(")[1][:-1]#calc_values(~tfdata,'speed',10,'average'~)~~の間を抽出
    function_params_list = function_params.split(",")
    function_params_list[1] = eval(function_params_list[1])
    function_params_list[3] = eval(function_params_list[3])

    #単位
    if use_unit_flag==True:

        if setV.check_domain(domain_class)=="mouse":
            distance_tanni = "cm"
            time_tanni = "s"
        elif setV.check_domain(domain_class)=="mouse_sm2":
            distance_tanni = "cm"
            time_tanni = "s"
        elif setV.check_domain(domain_class)=="senchu":
            distance_tanni = "mm"
            time_tanni = "s"
        elif setV.check_domain(domain_class)=="kokunusuto":
            distance_tanni = "mm"
            time_tanni = "s"
        elif setV.check_domain(domain_class)=="human":
            distance_tanni = ""
            time_tanni = ""
        else:
            distance_tanni = ""
            time_tanni = ""

        if distance_tanni+time_tanni!="":
            if function_params_list[1]=='speed':
                tanni_string = distance_tanni+"/"+time_tanni
            elif function_params_list[1]=='speed_acc':
                tanni_string = distance_tanni+"/"+time_tanni+"^2"
            else:
                tanni_string = "例外"
            if function_params_list[2]=='0' or function_params_list[3]=="average" or function_params_list[3]=="max" or function_params_list[3]=="min":
                tanni_string = "("+tanni_string+")"
            else:
                tanni_string = ""
        else:
            tanni_string = ""
    else:
        tanni_string = ""

    if function_params_list[2]=='0':
        return function_params_list[1].capitalize()+" "+tanni_string
    else:
        if function_params_list[3]=="waido":
            function_params_list[3]="skewness"
        elif function_params_list[3]=="sendo":
            function_params_list[3]="kurtosis"

        if function_params_list[1]=="speed_acc":
            function_params_list[1]="acceleration"

        return "Moving "+function_params_list[3]+" of "+function_params_list[1]+" [win size:"+str(int(function_params_list[2])*2)+"] "+tanni_string##ここ，windowsizehanを2倍して普通のwindowsizeを表記するようにしています！！


def get_cmap_colorbar_2(v_norm,cmap_name,min_v,max_v):

    #search color max
    pre_cmap_v = cmap_return(cmap_name,v_norm(max_v))
    for i in np.arange(max_v,min_v-(max_v-min_v)/100.,-(max_v-min_v)/100.):
        cmap_v = cmap_return(cmap_name,v_norm(i))
        if pre_cmap_v!=cmap_v:
            max_v = i
            break
        else:
            pre_cmap_v = cmap_v

    #search color min
    pre_cmap_v = cmap_return(cmap_name,v_norm(min_v))
    for i in np.arange(min_v,max_v+(max_v-min_v)/100.,(max_v-min_v)/100.):
        cmap_v = cmap_return(cmap_name,v_norm(i))
        if pre_cmap_v!=cmap_v:
            min_v = i
            break
        else:
            pre_cmap_v = cmap_v

    clist = []
    for i in np.arange(min_v,max_v+(max_v-min_v)/100.,(max_v-min_v)/100.):
        cmap_v = cmap_return(cmap_name,v_norm(i))
        clist.append(cmap_v)

    #print({'red':tuple(red_color),'green':tuple(green_color),'blue':tuple(blue_color)})

    return clist

def cmap_return(cmap_name,cmap_value):

    ##check##
    if cmap_value==1:
        cmap_value = 0.99

    ##make clear
    plane_cm = cm.seismic
    cm_list = plane_cm(np.arange(plane_cm.N))
    cm_list[:,-1] = 0.0
    clear_cmap = ListedColormap(cm_list)

    if cmap_name=="rainbow":
        if cmap_value>=0.5:
            return cm.rainbow(0.99)
        else:
            return cm.rainbow(cmap_value/0.5)
    elif cmap_name=="-rainbow":
        if cmap_value>=0.05:
            return cm.rainbow(0.0)
        else:
            return cm.rainbow(0.999-cmap_value/0.05)
    elif cmap_name=="Oranges":
        return cm.Oranges(cmap_value)
    elif cmap_name=="Reds":
        if cmap_value >= 1:
            return cm.Reds(0.99)
        return cm.Reds(cmap_value)
    elif cmap_name=="Maekawa":
        if cmap_value >= 1:
            return cm.get_cmap("autumn_r",128)(0.99)
        return cm.get_cmap("autumn_r",128)(cmap_value)
    elif cmap_name=="Maekawa_low":
        if cmap_value >= 0.6:
            return cm.get_cmap("autumn_r",128)(0.99)
        return cm.get_cmap("autumn_r",128)(cmap_value*5/3)
    elif cmap_name=="Maekawa_low2":
        if cmap_value >= 0.005:
            if (cmap_value-0.0015)*400 >= 0.99:
                return cm.get_cmap("autumn_r",128)(0.99)
            return cm.get_cmap("autumn_r",128)((cmap_value-0.0015)*400)
        else:
            return cm.get_cmap("autumn_r",128)(0.01)
    elif cmap_name=="Maekawa_low3":
        if cmap_value*55 >= 0.99:
            _color = list(cm.get_cmap("autumn_r",128)(0.99))
            _color[0] /= 1.13
            _color[1] /= 1.13
            _color[2] /= 1.13
            return tuple(_color)
        _color = list(cm.get_cmap("autumn_r",128)(cmap_value*55))
        _color[0] /= 1.13
        _color[1] /= 1.13
        _color[2] /= 1.13
        return tuple(_color)
    elif cmap_name=="Maekawa_low4":
        if cmap_value*110 >= 0.99:
            _color = list(cm.get_cmap("autumn_r",128)(0.99))
            _color[0] /= 1.13
            _color[1] /= 1.13
            _color[2] /= 1.13
            return tuple(_color)
        _color = list(cm.get_cmap("autumn_r",128)(cmap_value*110))
        _color[0] /= 1.13
        _color[1] /= 1.13
        _color[2] /= 1.13
        return tuple(_color)
    elif cmap_name=="Reds-grad":
        if cmap_value >= 0.002:
            if (cmap_value-0.0015)*300 >= 0.7:
                return cm.Reds(0.7)
            return cm.Reds((cmap_value-0.0015)*300)
        else:
            return cm.binary(0.38)
    elif cmap_name=="Reds-grey":
        if cmap_value >= 0.0022:
            return cm.Reds(0.7)
        else:
            return cm.binary(0.38)
    elif cmap_name=="coolwarm":
        return cm.coolwarm(cmap_value)
    elif cmap_name=="coolwarm-highpass":
        if cmap_value<=0.25:
            return cm.coolwarm(0)
        elif cmap_value>=0.75:
            return cm.coolwarm(0.99)
        elif cmap_value>0.25 and cmap_value<0.75:
            return cm.coolwarm(0.5)
    elif cmap_name=="coolwarm-highpass2":
        if cmap_value<=0.25:
            return cm.coolwarm(cmap_value)
        elif cmap_value>=0.75:
            return cm.coolwarm(cmap_value)
        elif cmap_value>0.25 and cmap_value<0.75:
            return cm.coolwarm(0.5)
    elif cmap_name=="red_gray":
        if cmap_value>0.5:
            return cm.Reds(0.75)#red
        elif cmap_value<=0.5:
            return cm.binary(0.38)#gray
    elif cmap_name=="red_blue_gray":
        if cmap_value>0.8:
            return cm.Reds(0.75)#red
        elif cmap_value<0.2:
            return cm.Blues(0.75)#blue
        else:
            return cm.binary(0.38)#gray
    elif cmap_name=="RGB":
        if cmap_value==0.99:
            return clear_cmap(0.1)#about white
        elif cmap_value==2:
            return cm.Oranges(0.5)#orange
        elif cmap_value==3:
            return cm.Blues(0.5)#blue
        elif cmap_value==4:
            return cm.binary(0.38)#gray
        else:
            print("error")
            print(cmap_value)
            sys.exit()
    elif cmap_name=="RGB2":
        if cmap_value==0.99:
            return cm.Blues(0.5)#blue
        elif cmap_value==2:
            return cm.Oranges(0.5)#orange
        elif cmap_value==0:
            return cm.binary(0.38)#gray
        else:
            print("error")
            print(cmap_value)
            sys.exit()
    elif cmap_name=="RdPu":
        return cm.RdPu(cmap_value)
    elif cmap_name=="PuRd":
        return cm.PuRd(cmap_value)
    elif cmap_name=="PuRd-sharp":
        if cmap_value>=0.8:
            return cm.PuRd(0.75)
        else:
            return cm.binary(0.5)
    else:
        return cm.cool(cmap_value)

def check_max_min(params_list,one_graph_feature):
    #get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list

    graph_feature_list_all = []
    #domain_class_list = os.listdir(activated_trajectory_dir+"/"+layer_node)
    for current_domain_class in setV.return_class_list(setV.check_domain(domain_class)):
        file_list = os.listdir(activated_trajectory_dir+"/"+layer_node+"/"+current_domain_class)
        for one_filename in file_list:

            #csv yomikomi
            #activated_trajectory
            activated_trajectory_data = ikeda_lib.read_csv(activated_trajectory_dir+"/"+layer_node+"/"+current_domain_class+"/"+one_filename)
            activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

            #trajectory
            trajectory_data = ikeda_lib.read_csv(trajectory_dir+"/"+current_domain_class+"/"+one_filename)
            trajectory_data = trajectory_data[1:].astype("float32")

            #trajectory_feature
            trajectory_feature_data = ikeda_lib.read_csv(trajectory_feature_dir+"/"+current_domain_class+"/"+one_filename)
            trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

            ##make set
            trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,one_filename]

            ##define params##
            plist = params_list
            tdatas = trajectory_data_set
            tdata = trajectory_data
            tfdata = trajectory_feature_data
            ###

            current_graph_feature = np.array([])
            current_params_list = [[layer_node,current_domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname]]
            current_graph_feature = eval(one_graph_feature)
            graph_feature_list_all.append(current_graph_feature)

    #calc normalize param
    graph_feature_list_all = np.array(graph_feature_list_all)

    return np.max(graph_feature_list_all), np.min(graph_feature_list_all)

###TODO###
#color no normalize
#html.py no hou no taiou
#
#
#@profile
def plot_images_gif(params_list,output_dir,value_normalization_class,valued_feature_func_name,cmap_name,check_max_min_flag,comment_memo):

    ##get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list

    #####settings#####
    image_dpi = 85#defalut:100
    #####settings#####

    ##output information memo##
    if os.path.exists(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information/"+output_dir.split("_")[-1]+".txt")==False:
        with open(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information/"+output_dir.split("/")[-1]+".txt", mode='w') as f:
            f.write(comment_memo)

    #####

    """if important_node_check_func(layer_node)==False:
        return"""

    ##check value_normalization
    value_norm = value_normalization_class(params_list,valued_feature_func_name)

    feature_max_min_dict = {}
    feature_max,feature_min = check_max_min(params_list,valued_feature_func_name)
    feature_mergin = (feature_max-feature_min)*0.02
    feature_max_min_dict["value"+":max"] = feature_max+feature_mergin
    feature_max_min_dict["value"+":min"] = feature_min-feature_mergin

    ##make dirs
    if os.path.exists(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory_gif")==False:
        os.makedirs(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory_gif")

    #filename_list = os.listdir(activated_trajectory_dir+"/"+layer_node+"/"+domain_class)
    ranking_list = ikeda_lib.read_csv(setV.analysis_directory+"/"+model_filename+"/class_predict_vector/"+domain_class+".csv")[1:,:]
    if setV.check_normal_dop(domain_class)=="normal":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,-1)[:,0]
    elif setV.check_normal_dop(domain_class)=="dop":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,1)[:,0]
    else:
        print("error in check_normal_dop")
        sys.exit()

    ##limit for make_image_num
    """if len(filename_list) > make_image_num:
        filename_list = filename_list[:make_image_num]"""

    ##make cmap for colorbar##
    colorbar = get_cmap_colorbar_2(value_norm.normalize_value,cmap_name,feature_max_min_dict["value:min"],feature_max_min_dict["value:max"])

    for filename in ranking_list[:8]:#[:8]

        #csv yomikomi
        #activated_trajectory
        activated_trajectory_data = ikeda_lib.read_csv(activated_trajectory_dir+"/"+layer_node+"/"+domain_class+"/"+filename)
        activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

        #trajectory
        trajectory_data = ikeda_lib.read_csv(trajectory_dir+"/"+domain_class+"/"+filename)
        trajectory_data = trajectory_data[1:].astype("float32")

        #trajectory_feature
        trajectory_feature_data = ikeda_lib.read_csv(trajectory_feature_dir+"/"+domain_class+"/"+filename)
        trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

        ##make set
        trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,filename]

        ###get valued_feature_list
        valued_feature_list = np.array([])
        plist = params_list
        tdatas = trajectory_data_set
        valued_feature_list = eval(valued_feature_func_name)
        ##normalize##
        #valued_feature_list = calc_moving_values(valued_feature_list,20,"average")

        ###output images

        ##original_trajectory
        plt.close()
        gs_master = GridSpec(nrows=1, ncols=2, width_ratios=[8, 1])
        fig = plt.figure()
        ##colorbar
        gs_cbar = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])
        ax_cbar = fig.add_subplot(gs_cbar[:,:])
        #ylist = np.arange(0.0, 1.0, 0.01)
        cmap2 = matlib.repmat(colorbar, 1, 2).reshape(-1, 2, 4)
        ax_cbar.imshow(cmap2, extent=[0, (feature_max_min_dict["value:max"]-feature_max_min_dict["value:min"])/20,feature_max_min_dict["value:max"],feature_max_min_dict["value:min"]])
        ax_cbar.tick_params(labelbottom=False, bottom=False)
        ax_cbar.tick_params(labeltop=False, top=False)
        ax_cbar.tick_params(labelleft=False, left=False)
        ax_cbar.tick_params(labelright=True, right=True)
        ax_cbar.set_xlabel("Attention")
        ax_cbar.invert_yaxis()
        ##figure
        gs_figure = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
        ax = plt.subplot(gs_figure[:,:])
        ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda x, loc: "{:,}".format(round(x,8)) if math.modf(round(x,8))[0]!=0.0 else "{:,}".format(int(x)) ) )#3桁区切り設定

        if use_unit_flag==True:
            if setV.check_domain(domain_class)=="mouse":
                tanni_string = "(cm)"
            elif setV.check_domain(domain_class)=="mouse_sm2":
                tanni_string = "(cm)"
            elif setV.check_domain(domain_class)=="senchu":
                tanni_string = "(mm)"
            elif setV.check_domain(domain_class)=="kokunusuto":
                tanni_string = "(mm)"
            elif setV.check_domain(domain_class)=="human":
                tanni_string = ""
            else:
                tanni_string = ""
        else:
            tanni_string = ""
        plt.xlabel('X '+tanni_string)
        plt.ylabel('Y '+tanni_string)
        append_gif_generator = []
        """
        xlim_range = (np.max(trajectory_data[:,3]) - np.min(trajectory_data[:,3]))/float(40)
        ylim_range = (np.max(trajectory_data[:,4]) - np.min(trajectory_data[:,4]))/float(40)
        plt.xlim(np.min(trajectory_data[:,3])-xlim_range,np.max(trajectory_data[:,3])+xlim_range)
        plt.ylim(np.min(trajectory_data[:,4])-ylim_range,np.max(trajectory_data[:,4])+ylim_range)"""
        xlim_range = (np.max(trajectory_data[:,3]) - np.min(trajectory_data[:,3]))
        x_mid = (np.max(trajectory_data[:,3]) + np.min(trajectory_data[:,3]))/2
        ylim_range = (np.max(trajectory_data[:,4]) - np.min(trajectory_data[:,4]))
        y_mid = (np.max(trajectory_data[:,4]) + np.min(trajectory_data[:,4]))/2

        if xlim_range >= ylim_range:
            plt.xlim(np.min(trajectory_data[:,3])-xlim_range/float(40),np.max(trajectory_data[:,3])+xlim_range/float(40))
            plt.ylim(y_mid-xlim_range/float(2)-xlim_range/float(40),y_mid+xlim_range/float(2)+xlim_range/float(40))
        else:
            plt.xlim(x_mid-ylim_range/float(2)-ylim_range/float(40),x_mid+ylim_range/float(2)+ylim_range/float(40))
            plt.ylim(np.min(trajectory_data[:,4])-xlim_range/float(40),np.max(trajectory_data[:,4])+xlim_range/float(40))

        memory_buf_trash = [0]*len(trajectory_data)
        for current_time_step in range(len(trajectory_data)):
            #plt.scatter(trajectory_data[current_time_step,3],trajectory_data[current_time_step,4],color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])), s=3)####点を描画するかどうか
            if current_time_step==0:
                continue
            plt.plot(trajectory_data[current_time_step-1:current_time_step+1,3],trajectory_data[current_time_step-1:current_time_step+1,4],color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])))
            if current_time_step%1==0:
                #plt.savefig(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory_gif"+"/trash/"+str(current_time_step)+".png",dpi=image_dpi)
                #append_gif_generator.append(Image.open(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory_gif"+"/trash/"+str(current_time_step)+".png"))
                #print(plt.gca())
                memory_buf_trash[current_time_step] = io.BytesIO()
                #
                plt.savefig(memory_buf_trash[current_time_step], format='png', dpi=image_dpi, bbox_inches='tight', pad_inches=0.1)
                memory_buf_trash[current_time_step].seek(0)
                append_gif_generator.append(Image.open(memory_buf_trash[current_time_step]))
                #memory_buf.close()

        append_gif_generator[0].save(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory_gif"+"/"+filename.split(".")[0]+".gif", save_all=True, append_images=append_gif_generator[1:],loop=0,duration=80)

        ##memory 対策#
        map(lambda memory_buf: memory_buf.close() ,memory_buf_trash)
        map(lambda ii: ii.close() ,append_gif_generator)

        plt.close()
        del memory_buf_trash
        del append_gif_generator
        gc.collect()
        ###

    return

def output_function_images(outname,function_name,params_list):

    ##get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list

    #####settings#####
    image_dpi = 85#defalut:100
    #####settings#####

    ##make dirs
    if os.path.exists(outname+"/"+layer_node+"/"+domain_class+"/"+function_name)==False:
        os.makedirs(outname+"/"+layer_node+"/"+domain_class+"/"+function_name)

    filename_list = os.listdir(activated_trajectory_dir+"/"+layer_node+"/"+domain_class)
    ##limit for make_image_num
    """if len(filename_list) > make_image_num:
        filename_list = filename_list[:make_image_num]"""
    ranking_list = ikeda_lib.read_csv(setV.analysis_directory+"/"+model_filename+"/class_predict_vector/"+domain_class+".csv")[1:,:]
    if setV.check_normal_dop(domain_class)=="normal":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,-1)[:,0]
    elif setV.check_normal_dop(domain_class)=="dop":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,1)[:,0]
    else:
        print("error in check_normal_dop")
        sys.exit()

    for filename in ranking_list[:output_number]:

        #csv yomikomi
        #activated_trajectory
        activated_trajectory_data = ikeda_lib.read_csv(activated_trajectory_dir+"/"+layer_node+"/"+domain_class+"/"+filename)
        activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

        exec(function_name+"(activated_trajectory_data)")
        plt.grid(which='major',color='grey',linestyle='--')
        plt.savefig(outname+"/"+layer_node+"/"+domain_class+"/"+function_name+"/"+filename.split(".")[0]+".svg",dpi=image_dpi, bbox_inches='tight', pad_inches=0.1)

    file = open(outname+"/"+layer_node+"/"+domain_class+"/"+function_name+'/feature_function.py', 'w')
    file.write(inspect.getsource(eval(function_name)))
    file.close()

    return

def plot_images(params_list,output_dir,value_normalization_class,valued_feature_func_name,cmap_name,check_max_min_flag,comment_memo):

    ##get params
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list

    #####settings#####
    ##graph_features
    graph_feature_names = [s for s in setV.setting_feature_name_list if "calc_values" in s]+ ["calc_values(tfdata,'speed',0,'average')"]#,"calc_values(tfdata,'speed',10,'variance')","calc_values(tfdata,'speed',10,'average')"
    image_dpi = 85#defalut:100
    #####settings#####

    ##output information memo##
    if os.path.exists(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information/"+output_dir.split("_")[-1]+".txt")==False:
        with open(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information/"+output_dir.split("/")[-1]+".txt", mode='w') as f:
            f.write(comment_memo)
    #####

    """if important_node_check_func(layer_node)==False:
        return"""

    ##check value_normalization
    value_norm = value_normalization_class(params_list,valued_feature_func_name)

    ##check features_max_min for plot_tatejiku
    feature_max_min_dict = {}
    for one_graph_feature in graph_feature_names:
        feature_max,feature_min = check_max_min(params_list,one_graph_feature)
        feature_mergin = (feature_max-feature_min)*0.02
        feature_max_min_dict[one_graph_feature+":max"] = feature_max+feature_mergin
        feature_max_min_dict[one_graph_feature+":min"] = feature_min-feature_mergin
    feature_max,feature_min = check_max_min(params_list,valued_feature_func_name)
    feature_mergin = (feature_max-feature_min)*0.02
    feature_max_min_dict["value"+":max"] = feature_max+feature_mergin
    feature_max_min_dict["value"+":min"] = feature_min-feature_mergin

    ##make dirs
    if os.path.exists(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory")==False:
        os.makedirs(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory")
    for one_graph_feature_name in graph_feature_names:
        if os.path.exists(output_dir+"/"+layer_node+"/"+domain_class+"/"+spTub(tr_display_name(one_graph_feature_name,None,False)))==False:
            os.makedirs(output_dir+"/"+layer_node+"/"+domain_class+"/"+spTub(tr_display_name(one_graph_feature_name,None,False)))
    if os.path.exists(output_dir+"/"+layer_node+"/"+domain_class+"/value")==False:
        os.makedirs(output_dir+"/"+layer_node+"/"+domain_class+"/value")

    filename_list = os.listdir(activated_trajectory_dir+"/"+layer_node+"/"+domain_class)
    ##limit for make_image_num
    """if len(filename_list) > make_image_num:
        filename_list = filename_list[:make_image_num]"""
    ranking_list = ikeda_lib.read_csv(setV.analysis_directory+"/"+model_filename+"/class_predict_vector/"+domain_class+".csv")[1:,:]
    if setV.check_normal_dop(domain_class)=="normal":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,-1)[:,0]
    elif setV.check_normal_dop(domain_class)=="dop":
        ranking_list = ikeda_lib.numpy_sort(ranking_list,1,1)[:,0]
    else:
        print("error in check_normal_dop")
        sys.exit()

    ##make cmap for colorbar##
    colorbar = get_cmap_colorbar_2(value_norm.normalize_value,cmap_name,feature_max_min_dict["value:min"],feature_max_min_dict["value:max"])
    #mappable = ScalarMappable(cmap=cmap_colorbar)
    #mappable._A = []

    for filename in ranking_list[:output_number]:

        #csv yomikomi
        #activated_trajectory
        activated_trajectory_data = ikeda_lib.read_csv(activated_trajectory_dir+"/"+layer_node+"/"+domain_class+"/"+filename)
        activated_trajectory_data = activated_trajectory_data[1:].astype("float32")

        #trajectory
        trajectory_data = ikeda_lib.read_csv(trajectory_dir+"/"+domain_class+"/"+filename)
        trajectory_data = trajectory_data[1:].astype("float32")

        #trajectory_feature
        trajectory_feature_data = ikeda_lib.read_csv(trajectory_feature_dir+"/"+domain_class+"/"+filename)
        trajectory_feature_data = trajectory_feature_data[1:].astype("float32")

        ##make set
        trajectory_data_set = [activated_trajectory_data,trajectory_data,trajectory_feature_data,filename]

        ###time feature
        timestamp_feature = trajectory_feature_data[:,0].T

        ###get valued_feature_list
        valued_feature_list = np.array([])
        plist = params_list
        tdatas = trajectory_data_set
        valued_feature_list = eval(valued_feature_func_name)
        ##normalize##
        #valued_feature_list = calc_moving_values(valued_feature_list,20,"average")

        ###output images

        ##original_trajectory
        plt.close()
        gs_master = GridSpec(nrows=1, ncols=2, width_ratios=[8, 1])
        gs_figure = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
        fig = plt.figure()
        ax = plt.subplot(gs_figure[:,:])
        ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda x, loc: "{:,}".format(round(x,8)) if math.modf(round(x,8))[0]!=0.0 else "{:,}".format(int(x)) ) )#3桁区切り設定

        for current_time_step in range(len(trajectory_data)):
            #plt.scatter(trajectory_data[current_time_step,3],trajectory_data[current_time_step,4],color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])), s=10)#,alpha=0.1点を描画するかどうか
            if current_time_step==0:
                continue
            plt.plot(trajectory_data[current_time_step-1:current_time_step+1,3],trajectory_data[current_time_step-1:current_time_step+1,4],color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])))

        if use_unit_flag==True:
            if setV.check_domain(domain_class)=="mouse":
                tanni_string = "(cm)"
            elif setV.check_domain(domain_class)=="mouse_sm2":
                tanni_string = "(cm)"
            elif setV.check_domain(domain_class)=="senchu":
                tanni_string = "(mm)"
            elif setV.check_domain(domain_class)=="kokunusuto":
                tanni_string = "(mm)"
            elif setV.check_domain(domain_class)=="human":
                tanni_string = ""
            else:
                tanni_string = ""
        else:
            tanni_string = ""
        plt.xlabel('X '+tanni_string)
        plt.ylabel('Y '+tanni_string)

        ##縦横比修正##
        xlim_range = (np.max(trajectory_data[:,3]) - np.min(trajectory_data[:,3]))
        x_mid = (np.max(trajectory_data[:,3]) + np.min(trajectory_data[:,3]))/2
        ylim_range = (np.max(trajectory_data[:,4]) - np.min(trajectory_data[:,4]))
        y_mid = (np.max(trajectory_data[:,4]) + np.min(trajectory_data[:,4]))/2

        if xlim_range >= ylim_range:
            plt.xlim(np.min(trajectory_data[:,3])-xlim_range/float(40),np.max(trajectory_data[:,3])+xlim_range/float(40))
            plt.ylim(y_mid-xlim_range/float(2)-xlim_range/float(40),y_mid+xlim_range/float(2)+xlim_range/float(40))
        else:
            plt.xlim(x_mid-ylim_range/float(2)-ylim_range/float(40),x_mid+ylim_range/float(2)+ylim_range/float(40))
            plt.ylim(np.min(trajectory_data[:,4])-xlim_range/float(40),np.max(trajectory_data[:,4])+xlim_range/float(40))
        ##

        gs_cbar = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])
        ax_cbar = fig.add_subplot(gs_cbar[:,:])
        #ylist = np.arange(0.0, 1.0, 0.01)
        cmap2 = matlib.repmat(colorbar, 1, 2).reshape(-1, 2, 4)
        ax_cbar.imshow(cmap2, extent=[0, (feature_max_min_dict["value:max"]-feature_max_min_dict["value:min"])/20,feature_max_min_dict["value:max"],feature_max_min_dict["value:min"]])
        ax_cbar.tick_params(labelbottom=False, bottom=False)
        ax_cbar.tick_params(labeltop=False, top=False)
        ax_cbar.tick_params(labelleft=False, left=False)
        ax_cbar.tick_params(labelright=True, right=True)
        ax_cbar.set_xlabel("Attention")
        ax_cbar.invert_yaxis()
        #ax_cbar.set_yscale('log')
        #plt.colorbar(mappable)
        plt.savefig(output_dir+"/"+layer_node+"/"+domain_class+"/trajectory"+"/"+filename.split(".")[0]+".svg",dpi=image_dpi, bbox_inches='tight', pad_inches=0.1)

        ##graph_features
        for one_graph_feature_name in graph_feature_names:
            current_graph_feature = np.array([])
            ##define params##
            pl = params_list
            tdata_set = trajectory_data_set
            tdata = trajectory_data
            tfdata = trajectory_feature_data
            ###
            current_graph_feature = eval(one_graph_feature_name)
            plt.close()
            gs_master = GridSpec(nrows=1, ncols=2, width_ratios=[8, 1])
            gs_figure = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
            fig = plt.figure()
            ax = plt.subplot(gs_figure[:,:])
            ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda x, loc: "{:,}".format(round(x,8)) if math.modf(round(x,8))[0]!=0.0 else "{:,}".format(int(x)) ) )#3桁区切り設定

            if check_max_min_flag!=None:
                plt.ylim(feature_max_min_dict[one_graph_feature_name+":min"], feature_max_min_dict[one_graph_feature_name+":max"])
            for current_time_step in range(len(trajectory_data)):
                if current_time_step==0:
                    continue
                plt.plot(timestamp_feature[current_time_step-1:current_time_step+1],np.array([current_graph_feature[current_time_step-1],current_graph_feature[current_time_step]]),color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])))

            if use_unit_flag==True:
                plt.xlabel('Time (s)')
            else:
                plt.xlabel('Time')
            plt.ylabel(tr_display_name(one_graph_feature_name,domain_class,use_unit_flag))

            gs_cbar = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])
            ax_cbar = fig.add_subplot(gs_cbar[:,:])
            #ylist = np.arange(0.0, 1.0, 0.01)
            cmap2 = matlib.repmat(colorbar, 1, 2).reshape(-1, 2, 4)
            ax_cbar.imshow(cmap2, extent=[0, (feature_max_min_dict["value:max"]-feature_max_min_dict["value:min"])/15,feature_max_min_dict["value:max"],feature_max_min_dict["value:min"]])
            ax_cbar.tick_params(labelbottom=False, bottom=False)
            ax_cbar.tick_params(labeltop=False, top=False)
            ax_cbar.tick_params(labelleft=False, left=False)
            ax_cbar.tick_params(labelright=True, right=True)
            ax_cbar.set_xlabel("Attention")
            ax_cbar.invert_yaxis()
            #plt.colorbar(mappable)
            plt.savefig(output_dir+"/"+layer_node+"/"+domain_class+"/"+spTub(tr_display_name(one_graph_feature_name,None,False))+"/"+filename.split(".")[0]+".svg",dpi=image_dpi, bbox_inches='tight', pad_inches=0.1)

        ##value_graph
        plt.close()
        if check_max_min_flag!=None:
            plt.ylim(feature_max_min_dict["value"+":min"], feature_max_min_dict["value"+":max"])
        for current_time_step in range(len(trajectory_data)):
            if current_time_step==0:
                continue
            plt.plot(timestamp_feature[current_time_step-1:current_time_step+1],np.array([valued_feature_list[current_time_step-1],valued_feature_list[current_time_step]]),color=cmap_return(cmap_name,value_norm.normalize_value(valued_feature_list[current_time_step])))

        if use_unit_flag==True:
            plt.xlabel('Time(s)')
        else:
            plt.xlabel('Time')
        plt.ylabel('Attention')
        plt.savefig(output_dir+"/"+layer_node+"/"+domain_class+"/value"+"/"+filename.split(".")[0]+".svg",dpi=image_dpi, bbox_inches='tight', pad_inches=0.1)
        #print("test5")

    return

def plot_make_function(params_list):
    [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[model_dirname,use_unit_flag] = params_list

    ##plot_images(param_list,outdir,normalization_class_name,,value_type,color_type,max_min_flag,comment)
    plot_images(params_list,output_image_dir+"/svg_output",not_normalization,"normal_attention1(plist,tdatas)","Maekawa_low4",None,"normal_attention1")

    return


if __name__ == '__main__':

    ###get filename from commandline###
    args = sys.argv
    if len(args)<=2:
        print("Usage:plot_highlight.py 'model_filename' 'original or normalize_type' 'only_node_name1' 'only_node_name2'...")
        sys.exit()
    model_filename = args[1]
    print(model_filename)
    normalize_type = args[2]
    print(normalize_type)

    ##get only nodenames
    if len(args)>3:
        only_node_names = args[3:]
    else:
        only_node_names = "ALL"

    ######

    ###instruction###
    #以下、描画するグラフに使用されているデータ元を表記#
    #trajectory# trajectory_dir

    #feature_value# trajectory_feature_dir

    #value# activated_trajectory_dir
    #なお，valueはコード内のnormalizeの設定によって、標準化する場合もある

    trajectory_dir = "trajectory"

    if normalize_type=="original":
        trajectory_feature_dir = "trajectory"
        use_unit_flag = True
    else:
        trajectory_feature_dir = "normalized_dataset/"+normalize_type+"/trajectory"
        use_unit_flag = False

    activated_trajectory_dir = setV.analysis_directory+"/"+model_filename+"/activated_trajectory"
    output_image_dir = setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory"
    ikeda_lib.make_dirs(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information")

    layer_node_dir_list = os.listdir(activated_trajectory_dir)
    domain_class_list = os.listdir(activated_trajectory_dir+"/"+layer_node_dir_list[0])

    shutil.copyfile(os.path.basename(__file__),setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/information/"+os.path.basename(__file__))

    ##make images
    print("Start to make images")
    pool_prosess = Pool(6)#

    for layer_node in layer_node_dir_list:
        ##skip node
        if only_node_names=="ALL":
            pass
        else:
            if layer_node not in only_node_names:
                continue

        print(layer_node)

        params_list = [[ [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[setV.analysis_directory+"/"+model_filename,use_unit_flag] ] for domain_class in domain_class_list]
        pool_prosess.map(plot_make_function, params_list)

        """for domain_class in domain_class_list:#for test
            params_list = [ [layer_node,domain_class],[trajectory_dir,trajectory_feature_dir,activated_trajectory_dir,output_image_dir],[setV.analysis_directory+"/"+model_filename,use_unit_flag] ]
            plot_make_function(params_list)"""
