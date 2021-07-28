from pdb import set_trace

import sys
import csv
import numpy as np

import os
import glob

import ikeda_lib

import setting_value as setV
from plot_highlight import tr_display_name,spTub

###get modelfilename from commandline###
args = sys.argv
if len(args)<=5:
    print("Usage:show_images-4.py 'model_filename' 'output_format' 'images_max' 'output_html_dir' 'image_dir1' 'image_dir2'...")
    sys.exit()
model_filename = args[1]
output_format = "."+args[2]
images_max = int(args[3])
output_html_dir = args[4]
print("model_filename:")
print(model_filename)

##get imagedirs
input_image_dirs = args[5:]
display_image_name = ""
for one_image_num in range(len(input_image_dirs)):
    display_image_name = display_image_name+"row"+str(one_image_num+1)+":"+input_image_dirs[one_image_num]+"/"
###########

###setting###

output_dir = setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory_view/"+output_html_dir

layer_node_lists = os.listdir(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/"+input_image_dirs[0])##activated_image no lstm infomation wo tukau

###
class_name_list = os.listdir(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/"+input_image_dirs[0]+"/"+layer_node_lists[0])
#feature_name_list = os.listdir(setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/"+input_image_dirs[0]+"/"+layer_node_lists[0]+"/"+class_name_list[0])
#print(feature_name_list)
feature_name_list = ["trajectory_gif"]+[spTub(tr_display_name(s,0,False)) for s in setV.setting_feature_name_list]+[spTub(tr_display_name("calc_values(tfdata,'speed',0,'average')",0,False)),"value"]#["trajectory_gif"]+ +["calc_values(tfdata,'speed',0,'average')","value"]
feature_name_display = "/\t".join(feature_name_list)
print("Using the following features")
print(feature_name_list)
#####

output_html_code = ""

for current_layer_node in layer_node_lists:

    ##html_code##
    output_html_code += "<p>"+current_layer_node+"</p>"
    #############

    DATA_SET_ALL = []
    for current_input_dir in input_image_dirs:
        layer_node = setV.analysis_directory+"/"+model_filename+"/highlighted_trajectory/"+current_input_dir+"/"+current_layer_node

        data_info = []
        if "trans-Short_A" in os.listdir(layer_node):
            data_info.append([layer_node+"/trans-Short_A",layer_node+"/trans-longA","kokunusu"])
        if "normal-pre" in os.listdir(layer_node):
            data_info.append([layer_node+"/normal-pre",layer_node+"/pd-pre","mouse"])
        if "normal-pre_sm2" in os.listdir(layer_node):
            data_info.append([layer_node+"/normal-pre_sm2",layer_node+"/pd-pre_sm2","mouse_sm2"])
        if "trans-norm" in os.listdir(layer_node):
            data_info.append([layer_node+"/trans-norm",layer_node+"/trans-dop3","senchu"])
        if "human_Co" in os.listdir(layer_node):
            data_info.append([layer_node+"/human_Co",layer_node+"/human_Pt","human"])

        data_num_max = len(os.listdir(data_info[0][0]+"/trajectory"))

        data_set = []
        for i in range(min(len(data_info),images_max)):

            sub_data_set1 = []

            for dir_name in feature_name_list:
                #通常
                """paths = os.listdir(data_info[i][0]+"/"+dir_name)
                paths.sort()"""
                #normal順ソート版
                ranking_list = ikeda_lib.read_csv(setV.analysis_directory+"/"+model_filename+"/class_predict_vector/"+data_info[i][0].split("/")[-1]+".csv")[1:,:]
                ranking_list = ikeda_lib.numpy_sort(ranking_list,1,-1)
                paths = list(map(lambda x: x[:-4], ranking_list[:,0]))
                display_params = list(map(lambda x: ["normal_prob:"+x[0],"dop_prob:"+x[1]], ranking_list[:,1:3]))
                ###
                true_path = []
                #gif画像か通常の画像か
                if dir_name.split("_")[-1]=="gif":
                    #gifなら
                    for j in range(len(paths)):
                        if os.path.exists(data_info[i][0]+"/"+dir_name+"/"+paths[j]+".gif"):
                            #gifがあれば表示
                            true_path.append([data_info[i][0]+"/"+dir_name+"/"+paths[j]+".gif",display_params[j]])
                        else:
                            #なければプレーン画像表示
                            true_path.append([data_info[i][0]+"/"+dir_name.split("_")[0]+"/"+paths[j]+output_format,display_params[j]])
                else:
                    #通常画像なら
                    for j in range(len(paths)):
                        true_path.append([data_info[i][0]+"/"+dir_name+"/"+paths[j]+output_format,display_params[j]])
                sub_data_set1.append(true_path)

            sub_data_set2 = []

            for dir_name in feature_name_list:
                #通常
                """paths = os.listdir(data_info[i][1]+"/"+dir_name)
                paths.sort()"""
                #dop順ソート版
                ranking_list = ikeda_lib.read_csv(setV.analysis_directory+"/"+model_filename+"/class_predict_vector/"+data_info[i][1].split("/")[-1]+".csv")[1:,:]
                ranking_list = ikeda_lib.numpy_sort(ranking_list,1,1)
                paths = list(map(lambda x: x[:-4], ranking_list[:,0]))
                display_params = list(map(lambda x: ["normal_prob:"+x[0],"dop_prob:"+x[1]], ranking_list[:,1:3]))
                ###
                true_path = []
                #gif画像か通常の画像か
                if dir_name.split("_")[-1]=="gif":
                    #gifなら
                    for j in range(len(paths)):
                        if os.path.exists(data_info[i][1]+"/"+dir_name+"/"+paths[j]+".gif"):
                            #gifがあれば表示
                            true_path.append([data_info[i][1]+"/"+dir_name+"/"+paths[j]+".gif",display_params[j]])
                        else:
                            #なければプレーン画像表示
                            true_path.append([data_info[i][1]+"/"+dir_name.split("_")[0]+"/"+paths[j]+output_format,display_params[j]])
                else:
                    #通常画像なら
                    for j in range(len(paths)):
                        true_path.append([data_info[i][1]+"/"+dir_name+"/"+paths[j]+output_format,display_params[j]])
                sub_data_set2.append(true_path)

            data_set.append([sub_data_set1,sub_data_set2,data_info[i][2]])

        DATA_SET_ALL.append(data_set)

    #gif trajectory#
    if os.path.exists(data_info[0][0]+"/trajectory_gif")==True:
        for domain_num in range(len(DATA_SET_ALL[0])):
            ##html_code##
            output_html_code += '<hr><p style="font-size: 90%;">'+'<u>Domain:'+DATA_SET_ALL[0][domain_num][2]+'(trajectory_gif)</u></p>'
            class_name = ["Normal","Dop"]
            for class_num in range(2):
                output_html_code += '<p style="font-size: 80%;">'+'['+class_name[class_num]+'Top trajectories (cls prob)]'#+feature_name_display+'<br>'
                output_html_code += '</p>'
                for valued_num in range(len(input_image_dirs)):
                    output_html_code += '<p style="font-size: 80%;">(Highlihgt value:'+input_image_dirs[valued_num]+')</p><span>'
                    for data_name in glob.glob(data_info[domain_num][class_num]+"/trajectory_gif/*.gif")[:4]:
                        output_html_code += '<div style="display:inline;position: relative;"><a href="../../../../'
                        output_html_code += (data_name)
                        output_html_code += '" target="_blank">'
                        output_html_code += '<img src="../../../../'
                        output_html_code += (data_name)
                        output_html_code += '" width="'+str(100.0/4)+'%"/>' + '</a></div>'
                    if valued_num!=len(input_image_dirs):
                        output_html_code += '<br><br><br>'
                    output_html_code += "</span>"
                output_html_code += '</p>'
                if class_num==0:
                    output_html_code += '<hr style="border-top:dashed 1px;height:1px;color:#FFFFFF;">'
    ##

    for data_num in range(data_num_max):

        ##html_code##
        #output_html_code += "<p>num:"+str(data_num)+"</p>"
        #############
        for domain_num in range(len(DATA_SET_ALL[0])):
            ##html_code##
            output_html_code += '<hr><p style="font-size: 90%;">'+'<u>Domain:'+DATA_SET_ALL[0][domain_num][2]+'</u></p>'
            class_name = ["Normal","Dop"]
            for class_num in range(2):
                output_html_code += '<p style="font-size: 80%;">'+'['+class_name[class_num]+']'#+feature_name_display+'<br>'
                for ii in range(len(DATA_SET_ALL[0][domain_num][class_num][0][data_num][1])):
                    output_html_code += DATA_SET_ALL[0][domain_num][class_num][0][data_num][1][ii]+'/'
                output_html_code += '</p>'
                for valued_num in range(len(input_image_dirs)):
                    output_html_code += '<p style="font-size: 80%;">(Highlihgt value(value):'+input_image_dirs[valued_num]+')</p><span>'
                    for feature_num in range(len(feature_name_list)):
                        output_html_code += '<div style="display:inline;position: relative;"><p style="position: absolute;top: 50%;left: 50%;transform: translate(-50%,-50%);font-size: 80%;">'+feature_name_list[feature_num]+'</p><a href="../../../../'
                        output_html_code += (DATA_SET_ALL[valued_num][domain_num][class_num][feature_num][data_num][0])
                        output_html_code += '" target="_blank">'
                        output_html_code += '<img src="../../../../'
                        output_html_code += (DATA_SET_ALL[valued_num][domain_num][class_num][feature_num][data_num][0])
                        output_html_code += '" width="'+str(100.0/len(feature_name_list))+'%"/>' + '</a></div>'
                    if valued_num!=len(input_image_dirs):
                        output_html_code += '<br><br><br>'
                    output_html_code += "</span>"
                output_html_code += '</p>'
                if class_num==0:
                    output_html_code += '<hr style="border-top:dashed 1px;height:1px;color:#FFFFFF;">'
            #############

    ##add header and footer
    output_html_code = '<!DOCTYPE html><html><head><meta http-equiv="X-UA-Compatible" content="IE=edge"><title></title><meta charset="utf-8"><meta name="description" content=""><meta name="author" content=""><meta name="viewport" content="width=device-width, initial-scale=1"><!--[if lt IE 9]><script src="//cdn.jsdelivr.net/html5shiv/3.7.2/html5shiv.min.js"></script><script src="//cdnjs.cloudflare.com/ajax/libs/respond.js/1.4.2/respond.min.js"></script><![endif]--><link rel="shortcut icon" href=""><style type="text/css"><!-- --></style></head><body>' + output_html_code + '</body></html>'

    ##write
    ikeda_lib.make_dirs(output_dir)
    writer = open(output_dir+"/"+current_layer_node+".html","w")
    writer.write(output_html_code)


###make main html
output_html_code = '<p>activated_images</p>'
layer_node_lists.sort()
for current_layer_node in layer_node_lists:
    output_html_code += '<p><a href="'
    output_html_code += current_layer_node+".html"
    output_html_code += '" target="_blank">'
    output_html_code += current_layer_node + '</a></p>'

output_html_code = '<!DOCTYPE html><html><head><meta http-equiv="X-UA-Compatible" content="IE=edge"><title></title><meta charset="utf-8"><meta name="description" content=""><meta name="author" content=""><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href=""><!--[if lt IE 9]><script src="//cdn.jsdelivr.net/html5shiv/3.7.2/html5shiv.min.js"></script><script src="//cdnjs.cloudflare.com/ajax/libs/respond.js/1.4.2/respond.min.js"></script><![endif]--><link rel="shortcut icon" href=""></head><body>' + output_html_code + '</body></html>'

##write
ikeda_lib.make_dirs(output_dir)
writer = open(output_dir+"/"+"main.html","w")
writer.write(output_html_code)

print("'file://"+os.path.dirname(os.path.abspath(__file__))+"/"+output_dir+"/"+"main.html'")
