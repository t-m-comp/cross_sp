import numpy as np
import os
import sys
import csv
import glob
import ikeda_lib

##machine learning library
import sklearn
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from sklearn.externals.six import StringIO

from dtreeviz.trees import *

import setting_value as setV
from time_feature_func_lib import *
from plot_highlight import tr_display_name

###input###
def encorder_angle_at(data_csv,normal_prob):

    return data_csv[:,7:11]

def encorder_speed_at(data_csv,normal_prob):

    return data_csv[:,15:19]

def encorder_angle_ft_xmean(data_csv,normal_prob):

    feature_label = data_csv[:1,11:15]
    attention_angle_vec = data_csv[1:,6].astype("float32")
    encorder_angle_ft_vec = data_csv[1:,11:15].astype("float32")

    return_vec = [0]*len(encorder_angle_ft_vec[0])
    for i in range(len(attention_angle_vec)):
        return_vec += encorder_angle_ft_vec[i]*attention_angle_vec[i]

    feature_label = np.vstack(( feature_label,np.array([return_vec]) ))

    return feature_label

def encorder_speed_ft_xmean(data_csv,normal_prob):

    feature_label = data_csv[:1,19:23]
    attention_speed_vec = data_csv[1:,5].astype("float32")
    encorder_speed_ft_vec = data_csv[1:,19:23].astype("float32")

    return_vec = [0]*len(encorder_speed_ft_vec[0])
    for i in range(len(attention_speed_vec)):
        return_vec += encorder_speed_ft_vec[i]*attention_speed_vec[i]

    feature_label = np.vstack(( feature_label,np.array([return_vec]) ))

    return feature_label

def various_feature_speed_ft_xmean(data_csv,normal_prob):

    vfeature_csv = various_feature(data_csv,normal_prob)

    feature_label = vfeature_csv[:1,:]
    attention_speed_vec = data_csv[1:,5].astype("float32")
    encorder_speed_ft_vec = vfeature_csv[1:,:].astype("float32")
    return_vec = [0]*len(encorder_speed_ft_vec[0])

    for i in range(len(attention_speed_vec)):
        return_vec += encorder_speed_ft_vec[i]*attention_speed_vec[i]

    feature_label = np.vstack(( feature_label,np.array([return_vec]) ))

    return feature_label

def various_feature(data_csv,normal_prob):

    tdata = data_csv[1:,:5].astype('float32')

    feature_matrix = []
    for value_type in ["speed","speed_acc"]:#"angle","angle_abs","angle_acc","angle_acc_abs"
        for calc_type in ["average","variance","waido","sendo","max","min"]:#
            for window_size_han in [0,5,10,25,50]:#
                current_name = tr_display_name("calc_values(tfdata,'"+value_type+"',"+str(window_size_han)+",'"+calc_type+"')",None,False)
                #current_name = value_type+"_"+calc_type+str(window_size_han*2)#2baini shitemasu!!!
                current_values = calc_values(tdata,value_type,window_size_han,calc_type)
                current_list = np.hstack(( np.array([current_name]), current_values ))
                feature_matrix.append(current_list)

    feature_matrix = np.array(feature_matrix).T

    return feature_matrix

###outpu###
def encorder_angle_ft2(data_csv,normal_prob):

    return data_csv[:,13]

def encorder_angle_ft3(data_csv,normal_prob):

    return data_csv[:,14]

def encorder_angle_at0(data_csv,normal_prob):

    return data_csv[:,7]

def attention_angle(data_csv,normal_prob):

    return data_csv[:,6]

def attention_speed(data_csv,normal_prob):

    return data_csv[:,5]

def attention_speed_high(data_csv,normal_prob):

    attention_speed_value = data_csv[1:,5].astype("float32")

    return_value = np.hstack(( ["none/attention"], np.where(attention_speed_value>(1/538.0),1,0).astype("str") ))

    return return_value

def attention_angle_high(data_csv,normal_prob):

    attention_angle_value = data_csv[1:,6].astype("float32")

    return_value = np.hstack(( ["none/attention"], np.where(attention_angle_value>(1/538.0),1,0).astype("str") ))

    return return_value

def normal_prob_return(data_csv,normal_prob):

    return np.array(["normal_prob",normal_prob])

def normal_prob_classify(data_csv,normal_prob):

    if normal_prob>=0.5:
        normal_prob = 1
    else:
        normal_prob = 0

    return np.array(["dop/normal",normal_prob])

def evaluation_func(dc_list,predict_vec_dir,activated_trajectory_dir,output_dir,type_list,max_depth):

    domain_name = dc_list[0]
    class_list = dc_list[1]
    input_type,output_type,eval_model_type = type_list

    #get predict label#
    predict_vec_csvlist = []
    for one_class in class_list:
        predict_vec_csvlist.append(ikeda_lib.read_csv(predict_vec_dir+"/"+one_class+".csv"))

    #get activated data#
    all_data_list = []
    for one_class in class_list:
        all_data_list += glob.glob(activated_trajectory_dir+"/"+one_class+"/*.csv")

    #get label#
    data_csv = ikeda_lib.read_csv(all_data_list[0])
    input_label = eval(input_type+"(data_csv,0).tolist()[0]")
    output_label = eval(output_type+"(data_csv,0).tolist()[0]")
    output_class_label = output_label.split("/")
    input_data_list = []
    output_data_list = []
    #get data
    for one_data in all_data_list:
        data_csv = ikeda_lib.read_csv(one_data)
        predict_vec_csv = predict_vec_csvlist[class_list.index(one_data.split("/")[-2])]
        #print(predict_vec_csv)
        normal_prob = float(predict_vec_csv[predict_vec_csv[:,0]==one_data.split("/")[-1]][0,1])

        input_data_list += eval(input_type+"(data_csv,normal_prob)[1:].astype('float32').tolist()")
        output_data_list += eval(output_type+"(data_csv,normal_prob)[1:].astype('float32').tolist()")
    #split data
    test_ratio = 0.2
    input_data_train = input_data_list[:-int(len(input_data_list)*test_ratio)]
    input_data_test = input_data_list[-int(len(input_data_list)*test_ratio):]
    output_data_train = output_data_list[:-int(len(output_data_list)*test_ratio)]
    output_data_test = output_data_list[-int(len(output_data_list)*test_ratio):]

    ##train param setting##
    max_depth = int(max_depth)

    #Train
    if eval_model_type=="regressor":
        clf = DecisionTreeRegressor(random_state=1,max_depth=max_depth)
        clf = clf.fit(np.array(input_data_train), np.array(output_data_train))

        #Test
        predicted = clf.predict(input_data_test)
        #Evaluation
        test_accuracy = np.average(np.abs(predicted - output_data_test)) / float(len(predicted))

        print(test_accuracy)

        #output tree structure
        dot_data = StringIO()
        sklearn.tree.export_graphviz(clf, out_file=dot_data,feature_names=input_label,filled=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        current_outdir = output_dir+"/"+layer_node+"/"+input_type+"/"+output_type+"/"+eval_model_type+"/Mdepth"+str(max_depth)
        ikeda_lib.make_dirs(current_outdir)
        graph.write_pdf(current_outdir+"/"+domain_name+"_Tacc"+str(test_accuracy)+".pdf")

    elif eval_model_type=="classifier":
        clf = DecisionTreeClassifier(random_state=1,max_depth=max_depth)
        clf = clf.fit(np.array(input_data_train), np.array(output_data_train))

        #Test
        predicted = clf.predict(input_data_test)
        #Evaluation
        test_accuracy = sum(predicted == output_data_test) / float(len(predicted))

        print(test_accuracy)

        #output tree structure
        viz = dtreeviz(clf,
               np.array(input_data_train),
               np.array([int(i) for i in output_data_train]),
               target_name='class',
               feature_names=input_label,
               class_names=output_class_label)
        current_outdir = output_dir+"/"+layer_node+"/"+input_type+"/"+output_type+"/"+eval_model_type+"/Mdepth"+str(max_depth)
        ikeda_lib.make_dirs(current_outdir)
        viz.save(current_outdir+"/"+domain_name+"_Tacc"+str(test_accuracy)+"_viz.svg")

    elif eval_model_type=="linearregressor":

        clf = LinearRegression()
        clf.fit(input_data_train,output_data_train)

        #Test
        linear_params = clf.coef_
        #Evaluation
        train_accuracy = clf.score(input_data_train, output_data_train)
        test_accuracy = clf.score(input_data_test, output_data_test)

        output_csv = np.array([input_label,linear_params]).T
        output_csv_sorted = output_csv[(((-1)*np.abs(output_csv[:,1].astype("float32"))).argsort()),:]

        current_outdir = output_dir+"/"+layer_node+"/"+input_type+"/"+output_type+"/"+eval_model_type
        ikeda_lib.make_dirs(current_outdir)
        ikeda_lib.write_csv(current_outdir+"/"+domain_name+"_R"+str(train_accuracy)+"_TR"+str(test_accuracy)+".csv",output_csv_sorted)

        print(input_label)
        print(linear_params)
        print(train_accuracy)
        print(test_accuracy)

    else:
        print("error in evaluation_func!!")
        sys.exit()

    return

def main_func(model_filename,layer_node,input_type,output_type,eval_model_type,max_depth):

    print("domainA/classA:")
    dA_cA = input()
    print("domainA/classB:")
    dA_cB = input()

    print("domainA_name:")
    dA_name = input()

    ##origin data
    predict_vec_dir =  setV.analysis_directory+"/"+model_filename+"/class_predict_vector"
    activated_trajectory_dir =  setV.analysis_directory+"/"+model_filename+"/activated_trajectory/"+layer_node

    ##for output
    output_dir = setV.analysis_directory+"/"+model_filename+"/dicisiontree_check"

    ##main
    domain_class_list = os.listdir(activated_trajectory_dir)

    ##prepare for exec dc list
    exec_dc_list = []
    #exec_dcall_list = ["all",[]]#ex. ["all",[["normal-pre","trans-Short_A","trans-norm"],["pd-pre","trans-longA","trans-dop3"]]]

    exec_dc_list.append([dA_name,[dA_cA,dA_cB]])
    #exec_dcall_list[1].append(dA_cA)
    #exec_dcall_list[1].append(dA_cB)

    #exec_dc_list.append(exec_dcall_list)

    ##exec for each domain
    for dc_list in exec_dc_list:
        evaluation_func(dc_list,predict_vec_dir,activated_trajectory_dir,output_dir,[input_type,output_type,eval_model_type],max_depth)

    return

if __name__ == '__main__':

    ###get params from commandline###
    args = sys.argv
    if len(args)!=7:
        print("Usage:dicision_tree_check.py 'model_filename' 'layer_node' 'input_type' 'output_type' 'eval_model_type' 'max_depth'")
        sys.exit()
    model_filename = args[1]
    layer_node = args[2]
    input_type = args[3]
    output_type = args[4]
    eval_model_type = args[5]
    max_depth = args[6]
    ######

    main_func(model_filename,layer_node,input_type,output_type,eval_model_type,max_depth)
