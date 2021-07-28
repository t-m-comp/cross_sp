from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from flip_gradient import flip_gradient
from utils import *

import csv
import os

import pdb
from datetime import datetime

import setting_value as setV
import ikeda_lib

import sys

from twodomain_train import build_model, get_train_and_test_datas

input_row = 2
batch_size = 18
time_steps = 538

def write_information_data(model_filename,csv_name,numpy_data):

    #csv kakikomi
    ikeda_lib.make_dirs(setV.analysis_directory+"/"+model_filename+"/information_dir")
    flout  = open(setV.analysis_directory+"/"+model_filename+"/information_dir/"+csv_name+".csv","w")
    writer = csv.writer(flout,lineterminator="\n")
    writer.writerows(numpy_data)
    flout.close()

    return

def get_datas(dataset_dir):
    Xs = []
    filename_list = []
    data_list = sorted(os.listdir(dataset_dir))
    for one_file_num in range(len(data_list)):
        flin = open(dataset_dir+"/"+data_list[one_file_num],"r")
        dataReader = csv.reader(flin)
        csv_data = []
        for line in dataReader:
            csv_data.append(line)
        flin.close()
        Xs.append(csv_data[1:][:])
        filename_list.append(data_list[one_file_num])

    Xs = np.array(Xs).astype("float32")
    filename_list = np.array(filename_list)

    return Xs[:],filename_list[:]#Xs[:32],filename_list[:32]

def train_and_evaluate(sess,data_list_params,dir_name_params,normalize_dir_name):

    [data_list_all,filename_list_all,tt_data_list_all] = data_list_params
    [domain_class_trajectory_dir,output_dirname] = dir_name_params

    ##get datas and params##
    [[X_a,y_a,X_aT, y_aT]] = tt_data_list_all
    [[X_dA_cA,X_dA_cB]] = data_list_all

    # Get output tensors
    p_acc = sess.graph.get_tensor_by_name('p_acc:0')

    ######GET ATTENTION#######
    attented_weights_1 = sess.graph.get_tensor_by_name('attented_weights_speed:0')
    attented_weights_2 = sess.graph.get_tensor_by_name('attented_weights_angle:0')
    encorder_output_angle_at = sess.graph.get_tensor_by_name('encorder_output_angle_at:0')
    encorder_output_angle_ft = sess.graph.get_tensor_by_name('encorder_output_angle_ft:0')
    encorder_output_speed_at = sess.graph.get_tensor_by_name('encorder_output_speed_at:0')
    encorder_output_speed_ft = sess.graph.get_tensor_by_name('encorder_output_speed_ft:0')
    ######GET ATTENTION#######

    class_predict_vector = sess.graph.get_tensor_by_name('class_predict_vector:0')

    ## Get final accuracies on whole dataset & lstm_output_list

    #train_accuracy
    pa_A = sess.run([p_acc], feed_dict={'X:0': X_a,'Y_ind:0': y_a, 'keep_prob:0':1.0})
    print("*train_accuracy*")
    print('Aclass: ', pa_A)
    #test_accuracy
    pa_A = sess.run([p_acc], feed_dict={'X:0': X_aT,'Y_ind:0': y_aT, 'keep_prob:0':1.0})
    print("*test_accuracy*")
    print('Aclass: ', pa_A)
    #hole dataset accuracy
    pa_A = sess.run([p_acc], feed_dict={'X:0': np.vstack((X_a,X_aT)),'Y_ind:0': np.hstack((y_a,y_aT)), 'keep_prob:0':1.0})
    print("*all_dataset_accuracy*")
    print('Aclass: ', pa_A)

    print("output ok?(y/n):")
    output_ok = input()
    if output_ok!="y":
        #print("fin")
        sys.exit()
    else:
        print("ok")

    ###get class_predict_vector###
    class_predict_vector_dir = output_dirname+"/class_predict_vector"
    ikeda_lib.make_dirs(class_predict_vector_dir)
    for domain_num in range(1):
        for class_num in range(2):
            class_vector_value = sess.run([class_predict_vector], feed_dict={'X:0': data_list_all[domain_num][class_num], 'keep_prob:0':1.0})
            output_np = np.hstack((filename_list_all[domain_num][class_num].reshape(-1,1),class_vector_value[0]))
            output_np = np.vstack((np.array(["filename","normal","dop"]),output_np))
            ikeda_lib.write_csv(class_predict_vector_dir+"/"+domain_class_trajectory_dir[domain_num][class_num]+".csv",output_np)

    print("class_predict-OK")
    ##############################

    ###get attention###
    class_number = len(domain_class_trajectory_dir[0])

    for domain_num in range(1):
        for class_num in range(class_number):
            activate_dir_name = "attention-layer"
            if os.path.exists(output_dirname+"/activated_trajectory/"+activate_dir_name+"/"+domain_class_trajectory_dir[domain_num][class_num])==False:
                os.makedirs(output_dirname+"/activated_trajectory/"+activate_dir_name+"/"+domain_class_trajectory_dir[domain_num][class_num])

            #get attention weights
            [attented_weights_value_1,attented_weights_value_2] = sess.run([attented_weights_1,attented_weights_2], feed_dict={'X:0': data_list_all[domain_num][class_num], 'keep_prob:0':1.0})
            [encorder_output_angle_at_V,encorder_output_angle_ft_V,encorder_output_speed_at_V,encorder_output_speed_ft_V] = sess.run([encorder_output_angle_at,encorder_output_angle_ft,encorder_output_speed_at,encorder_output_speed_ft], feed_dict={'X:0': data_list_all[domain_num][class_num], 'keep_prob:0':1.0})

            for data_num in range(len(attented_weights_value_1)):
                current_filename = filename_list_all[domain_num][class_num][data_num]
                #csv yomikomi
                trajectory_data = ikeda_lib.read_csv("trajectory/"+domain_class_trajectory_dir[domain_num][class_num]+"/"+current_filename)
                feature_data = ikeda_lib.read_csv(normalize_dir_name+"/trajectory/"+domain_class_trajectory_dir[domain_num][class_num]+"/"+current_filename)

                ##add trajectory and attention
                #make data part
                output_data_part = np.hstack((trajectory_data[1:,:1],feature_data[1:,1:3],trajectory_data[1:,3:5],attented_weights_value_1[data_num,:,:],attented_weights_value_2[data_num,:,:],encorder_output_angle_at_V[data_num,:,:],encorder_output_angle_ft_V[data_num,:,:],encorder_output_speed_at_V[data_num,:,:],encorder_output_speed_ft_V[data_num,:,:]))

                #make header part
                _buf = ["time","angle","speed","x","y","attention_speed","attention_angle"]
                for main_label in ["encorder_angle_at"]:
                    for cell_num in range(len(encorder_output_angle_at_V[0][0])):
                        _buf.append(main_label+str(cell_num))
                for main_label in ["encorder_angle_ft"]:
                    for cell_num in range(len(encorder_output_angle_ft_V[0][0])):
                        _buf.append(main_label+str(cell_num))
                for main_label in ["encorder_speed_at"]:
                    for cell_num in range(len(encorder_output_speed_at_V[0][0])):
                        _buf.append(main_label+str(cell_num))
                for main_label in ["encorder_speed_ft"]:
                    for cell_num in range(len(encorder_output_speed_ft_V[0][0])):
                        _buf.append(main_label+str(cell_num))
                output_header_part = np.array([_buf])
                output_data = np.vstack((output_header_part,output_data_part))

                #csv kakikomi
                ikeda_lib.write_csv(output_dirname+"/activated_trajectory/"+activate_dir_name+"/"+domain_class_trajectory_dir[domain_num][class_num]+"/"+current_filename,output_data)

    print("attention-OK")

def main_fnc(model_filename,model_stepname,dA_cA,dA_cB,normalize_dir_name):

    ##input model name##
    """print("please type model filename:")
    model_filename = input()

    print("please type model stepname:")
    model_stepname = input()

    print("normal_class:")
    dA_cA = input()
    print("dop_class:")
    dA_cB = input()

    print("dataset_dir_name:")
    normalize_dir_name = input()"""

    ###get datasets(original)###
    ##domainA
    X_dA_cA,filename_list_dA_cA = get_datas(normalize_dir_name+"/dataset/"+dA_cA)
    X_dA_cB,filename_list_dA_cB = get_datas(normalize_dir_name+"/dataset/"+dA_cB)

    ##data_list_all##
    data_list_all = [[X_dA_cA,X_dA_cB]]
    ##filename_list_all
    filename_list_all = [[filename_list_dA_cA,filename_list_dA_cB]]

    ###get datasets(train_and_test)###
    ##domainA
    X_a,y_a,X_aT, y_aT,_buf,_buf2 = get_train_and_test_datas([normalize_dir_name+"/dataset/"+dA_cA,normalize_dir_name+"/dataset/"+dA_cB],0.2)

    ##train_and_test(tt)_data_list##
    tt_data_list_all = [[X_a,y_a,X_aT, y_aT]]

    ##domain_class_dir_name
    domain_class_trajectory_dir = [[dA_cA,dA_cB]]

    ##output dirname##
    output_dirname = setV.analysis_directory + "/"+model_filename+"_"+model_stepname

    #sess.close()
    tf.reset_default_graph()

    build_model()
    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, "model/"+model_filename+"/"+model_stepname)

    train_and_evaluate(sess,[data_list_all,filename_list_all,tt_data_list_all],[domain_class_trajectory_dir,output_dirname],normalize_dir_name)
    return

if __name__ == '__main__':

    ###get params from commandline###
    args = sys.argv
    if len(args)!=6:
        print("Usage:get_attention.py 'model_filename' 'model_stepname' 'dataset_dir' 'dA_cA' 'dA_cB'")
        sys.exit()
    model_filename = args[1]
    model_stepname = "step-"+args[2]
    normalize_dir_name = args[3]
    dA_cA = args[4]
    dA_cB = args[5]
    ######

    main_fnc(model_filename,model_stepname,dA_cA,dA_cB,normalize_dir_name)
