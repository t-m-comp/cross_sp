
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf##
import numpy as np
import pickle as pkl
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA

from flip_gradient import flip_gradient
from utils import *

import csv
import os
import sys
import shutil
import ikeda_lib

import pdb
from datetime import datetime

##define
input_row = 2
batch_size = 18
time_steps = 538
########
##model params#
ft_ex1 = 7
ft_ex2 = 4
atv = 3
dcv = 4
dloss_ats_x = 1
filter_height_one = 20
filter_height_two = 50
########

def get_train_and_test_datas(dataset_dir_list,test_wariai):
    Xs = []
    Xs_filename = []
    ys = []
    Xs_T = []
    Xs_T_filename = []
    ys_T = []
    for dataset_dir_num in range(len(dataset_dir_list)):
        data_list = sorted(os.listdir(dataset_dir_list[dataset_dir_num]))
        train_data_list = data_list[:int(len(data_list)*(1-test_wariai))]
        test_data_list = data_list[int(len(data_list)*(1-test_wariai)):]
        for one_file_num in range(len(train_data_list)):
            flin = open(dataset_dir_list[dataset_dir_num]+"/"+train_data_list[one_file_num],"r")
            dataReader = csv.reader(flin)
            csv_data = []
            for line in dataReader:
                csv_data.append(line)
            flin.close()
            Xs.append(csv_data[1:][:])
            Xs_filename.append(dataset_dir_list[dataset_dir_num]+"/"+train_data_list[one_file_num])
            ys.append(dataset_dir_num)

        for one_file_num in range(len(test_data_list)):
            flin = open(dataset_dir_list[dataset_dir_num]+"/"+test_data_list[one_file_num],"r")
            dataReader = csv.reader(flin)
            csv_data = []
            for line in dataReader:
                csv_data.append(line)
            flin.close()
            Xs_T.append(csv_data[1:][:])
            Xs_T_filename.append(dataset_dir_list[dataset_dir_num]+"/"+test_data_list[one_file_num])
            ys_T.append(dataset_dir_num)

    Xs = np.array(Xs).astype("float32")
    Xs_filename = np.array(Xs_filename)
    ys = np.array(ys)

    Xs_T = np.array(Xs_T).astype("float32")
    Xs_T_filename = np.array(Xs_T_filename)
    ys_T = np.array(ys_T)

    return Xs,ys,Xs_T,ys_T,Xs_filename,Xs_T_filename

def build_model():
    X = tf.placeholder(tf.float32, [None,time_steps,input_row], name='X') # Input data
    Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
    D_ind = tf.placeholder(tf.int32, [None], name='D_ind')  # Domain index
    l = tf.placeholder(tf.float32, [], name='l')        # Gradient reversal scaler
    train_aria = tf.placeholder(tf.string, [], name='train_aria')    #alltrain or only domainclassifier train
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')##dropout

    Y = tf.one_hot(Y_ind, 2)
    D = tf.one_hot(D_ind, 2)

    ###split input into speed and angle
    X_angle = X[:,:,:1]
    X_speed = X[:,:,1:]

    ###Feature extractor
    filter_height_one = 10
    filter_height_two = 40

    ##angle process
    X_angle = tf.reshape(X_angle,(-1,time_steps,1,1))
    #for attention
    W_conv1a = weight_variable([filter_height_one,1, 1,5])
    b_conv1a = bias_variable([5])
    W_conv1a_out = tf.nn.relu(tf.nn.conv2d(X_angle, W_conv1a, strides=[1,1,1,1], padding='SAME') + b_conv1a)
    W_conv1a_out = tf.nn.dropout(W_conv1a_out, keep_prob)
    W_conv2a = weight_variable([filter_height_two,1, 5,4])
    b_conv2a = bias_variable([4])
    W_conv2a_out = tf.nn.relu(tf.nn.conv2d(W_conv1a_out, W_conv2a, strides=[1,1,1,1], padding='SAME') + b_conv2a)
    W_conv2a_out = tf.nn.dropout(W_conv2a_out, keep_prob)
    encorder_output_angle_at = tf.reshape(W_conv2a_out,(-1,time_steps,4))##definition
    _buf = tf.add(encorder_output_angle_at,0,name="encorder_output_angle_at")

    #for feature extract
    W_conv3a = weight_variable([filter_height_one,1, 1,5])
    b_conv3a = bias_variable([5])
    W_conv3a_out = tf.nn.relu(tf.nn.conv2d(X_angle, W_conv3a, strides=[1,1,1,1], padding='SAME') + b_conv3a)
    W_conv3a_out = tf.nn.dropout(W_conv3a_out, keep_prob)
    W_conv4a = weight_variable([filter_height_two,1, 5,4])
    b_conv4a = bias_variable([4])
    W_conv4a_out = tf.nn.relu(tf.nn.conv2d(W_conv3a_out, W_conv4a, strides=[1,1,1,1], padding='SAME') + b_conv4a)
    W_conv4a_out = tf.nn.dropout(W_conv4a_out, keep_prob)
    encorder_output_angle_ft = tf.reshape(W_conv4a_out,(-1,time_steps,4))##definition
    _buf = tf.add(encorder_output_angle_ft,0,name="encorder_output_angle_ft")

    ##attention!!!
    #attention variables
    W_att1a = weight_variable([4, 3])
    W_att2a = weight_variable([3, 1])

    attented_weights_angle = tf.scan(lambda a,b: tf.nn.tanh(tf.matmul(b,W_att1a)) ,encorder_output_angle_at,initializer=tf.constant(0.0, shape=[time_steps, 3]))
    attented_weights_angle_presoftmax = tf.scan(lambda a,b: tf.matmul(b,W_att2a) ,attented_weights_angle,initializer=tf.constant(0.0, shape=[time_steps, 1]))
    attented_weights_angle = tf.scan(lambda a,b: tf.nn.softmax(b,axis=0) ,attented_weights_angle_presoftmax,initializer=tf.constant(0.0, shape=[time_steps, 1]))

    attented_weights_angle_t = tf.reshape(attented_weights_angle,(-1,1,time_steps))
    _buf = tf.add(attented_weights_angle,0,name="attented_weights_angle")

    attented_sum_angle = tf.matmul(attented_weights_angle_t,encorder_output_angle_ft)

    attention_output_angle = tf.reshape(attented_sum_angle,(-1,4))
    attention_output_angle = tf.nn.relu(attention_output_angle)
    #attention_output = tf.nn.dropout(attention_output, keep_prob)

    ##speed process
    with tf.name_scope('atfeautre_ectractor'):
        #for attention
        X_speed = tf.reshape(X_speed,(-1,time_steps,1,1))
        W_conv1s = weight_variable([filter_height_one,1, 1,ft_ex1])
        b_conv1s = bias_variable([ft_ex1])
        W_conv1s_out = tf.nn.relu(tf.nn.conv2d(X_speed, W_conv1s, strides=[1,1,1,1], padding='SAME') + b_conv1s)
        W_conv1s_out = tf.nn.dropout(W_conv1s_out, keep_prob)
        W_conv2s = weight_variable([filter_height_two,1, ft_ex1,ft_ex2])
        b_conv2s = bias_variable([ft_ex2])
        W_conv2s_out = tf.nn.relu(tf.nn.conv2d(W_conv1s_out, W_conv2s, strides=[1,1,1,1], padding='SAME') + b_conv2s)
        W_conv2s_out = tf.nn.dropout(W_conv2s_out, keep_prob)
        encorder_output_speed_at = tf.reshape(W_conv2s_out,(-1,time_steps,ft_ex2))##definition
    _buf = tf.add(encorder_output_speed_at,0,name="encorder_output_speed_at")

    with tf.name_scope('clfeature_ectractor'):
        #for feature extract
        W_conv3s = weight_variable([filter_height_one,1, 1,ft_ex1])
        b_conv3s = bias_variable([ft_ex1])
        W_conv3s_out = tf.nn.relu(tf.nn.conv2d(X_speed, W_conv3s, strides=[1,1,1,1], padding='SAME') + b_conv3s)
        W_conv3s_out = tf.nn.dropout(W_conv3s_out, keep_prob)
        W_conv4s = weight_variable([filter_height_two,1, ft_ex1,ft_ex2])
        b_conv4s = bias_variable([ft_ex2])
        W_conv4s_out = tf.nn.relu(tf.nn.conv2d(W_conv3s_out, W_conv4s, strides=[1,1,1,1], padding='SAME') + b_conv4s)
        W_conv4s_out = tf.nn.dropout(W_conv4s_out, keep_prob)
        encorder_output_speed_ft = tf.reshape(W_conv4s_out,(-1,time_steps,ft_ex2))##definition
    _buf = tf.add(encorder_output_speed_ft,0,name="encorder_output_speed_ft")

    ##attention!!!
    #attention variables
    W_att1s = weight_variable([ft_ex2, atv])
    W_att2s = weight_variable([atv, 1])

    attented_weights_speed = tf.scan(lambda a,b: tf.nn.tanh(tf.matmul(b,W_att1s)) ,encorder_output_speed_at,initializer=tf.constant(0.0, shape=[time_steps, atv]))
    attented_weights_speed_presoftmax = tf.scan(lambda a,b: tf.matmul(b,W_att2s) ,attented_weights_speed,initializer=tf.constant(0.0, shape=[time_steps, 1]))
    attented_weights_speed = tf.scan(lambda a,b: tf.nn.softmax(b,axis=0) ,attented_weights_speed_presoftmax,initializer=tf.constant(0.0, shape=[time_steps, 1]))

    attented_weights_speed_t = tf.reshape(attented_weights_speed,(-1,1,time_steps))
    _buf = tf.add(attented_weights_speed,0,name="attented_weights_speed")

    attented_sum_speed = tf.matmul(attented_weights_speed_t,encorder_output_speed_ft)

    attention_output_speed = tf.reshape(attented_sum_speed,(-1,ft_ex2))
    attention_output_speed = tf.nn.relu(attention_output_speed)

    ##stack feature vector
    #attention_output = tf.reshape(tf.stack([attention_output_angle,attention_output_speed],1),[-1,4*2])
    attention_output = attention_output_speed

    ##decorder
    W11 = weight_variable([ft_ex2, dcv])
    b11 = bias_variable([dcv])
    f_logit = tf.matmul(attention_output, W11) + b11
    feature_vector = tf.nn.relu(f_logit)
    feature_vector = tf.nn.dropout(feature_vector, keep_prob)

    W1 = weight_variable([dcv, 2])
    b1 = bias_variable([2])
    p_logit = tf.matmul(feature_vector, W1) + b1
    p = tf.nn.softmax(p_logit)
    not_used_buf = tf.add(p, 0, name="class_predict_vector")

    p_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=Y)

    ###Domain predictor
    encorder_output_speed_at_flip = flip_gradient(encorder_output_speed_at, l)
    attention_output_flip = flip_gradient(attention_output, l)


    with tf.name_scope('domain_classifier'):

        #encorder_output_angle_flip_rs = tf.reshape(encorder_output_angle_flip,(-1,time_steps*4))
        #encorder_output_speed_flip_rs = tf.reshape(encorder_output_speed_flip,(-1,time_steps*4))

        ##decorder
        ##準備D##
        D_timesteps = tf.scan(lambda a,b: tf.scan(lambda at,bt: b, tf.constant([[0]]*time_steps),initializer=tf.constant(0.0,shape=[2])), D, initializer=tf.constant(0.0, shape=[time_steps,2]))

        #attention feature

        W_att1at_d = weight_variable([ft_ex2, 3])
        W_att2at_d = weight_variable([3, 2])

        attented_weights_speed_atD = tf.scan(lambda a,b: tf.nn.tanh(tf.matmul(b,W_att1at_d)) ,encorder_output_speed_at_flip,initializer=tf.constant(0.0, shape=[time_steps, 3]))
        attented_weights_speed_atD = tf.scan(lambda a,b: tf.nn.softmax(tf.matmul(b,W_att2at_d),axis=1) ,attented_weights_speed_atD,initializer=tf.constant(0.0, shape=[time_steps, 2]))
        d_loss_at_speed = tf.reduce_mean(tf.square(D_timesteps-attented_weights_speed_atD))

        #domain classifier
        W_att1_d = weight_variable([ft_ex2, 3])
        W_att2_d = weight_variable([3, 2])

        attention_output_flip_d = tf.nn.tanh(tf.matmul(attention_output_flip,W_att1_d))
        attention_output_flip_d = tf.nn.softmax(tf.matmul(attention_output_flip_d,W_att2_d),axis=1)
        d_loss_xmean = tf.reduce_mean(tf.square(D-attention_output_flip_d))

    # Optimization
    pred_loss = tf.reduce_sum(p_loss, name='pred_loss')
    domain_loss_atspeed = tf.reduce_sum(tf.multiply(d_loss_at_speed,1), name='domain_loss_atspeed')
    domain_loss_xmean = tf.reduce_sum(tf.multiply(d_loss_xmean,1), name='domain_loss_xmean')
    domain_loss = tf.add(domain_loss_atspeed, domain_loss_xmean, name='domain_loss')
    #total_loss = tf.add(pred_loss, domain_loss, name='total_loss')

    #for dann_train_op
    pred_train_op = tf.train.AdamOptimizer().minimize(pred_loss, name='pred_train_op')
    pred_at_train_op = tf.train.AdamOptimizer().minimize(tf.add(pred_loss,tf.multiply(domain_loss_atspeed,dloss_ats_x)), name='pred_at_train_op')
    #domain_train_op = tf.train.AdamOptimizer().minimize(domain_loss, name='domain_train_op')
    #dann_train_op = tf.train.AdamOptimizer().minimize(total_loss, name='dann_train_op')

    #for domain_only_train_op
    domain_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="domain_classifier")
    domain_only_train_op_atspeed = tf.train.AdamOptimizer().minimize(domain_loss_atspeed, name='domain_only_train_op_atspeed',var_list=domain_train_vars)
    domain_only_train_op_xmean = tf.train.AdamOptimizer().minimize(domain_loss_xmean, name='domain_only_train_op_xmean',var_list=domain_train_vars)
    domain_only_train_op = tf.train.AdamOptimizer().minimize(domain_loss, name='domain_only_train_op',var_list=domain_train_vars)

    #for clfeature_domain_lp_op
    atfeature_domain_lp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="atfeautre_ectractor")
    atfeature_domain_lp_op = tf.train.AdamOptimizer().minimize(domain_loss_atspeed, name='atfeature_domain_lp_op',var_list=(atfeature_domain_lp_vars+domain_train_vars))
    clfeature_domain_lp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="clfeature_ectractor")
    clfeature_domain_lp_op = tf.train.AdamOptimizer().minimize(domain_loss_xmean, name='clfeature_domain_lp_op',var_list=(clfeature_domain_lp_vars+domain_train_vars))

    # Evaluation
    p_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(p, 1)), tf.float32), name='p_acc')
    d_xm_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D, 1), tf.argmax(attention_output_flip_d, 1)), tf.float32), name='d_xm_acc')##
    d_at_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reshape(D_timesteps,(-1,2)), 1), tf.argmax(tf.reshape(attented_weights_speed_atD,(-1,2)), 1)), tf.float32), name='d_at_acc')##
    _buf = tf.add(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(p, 1)), tf.int32),0,name='eval_acc')

    ###weights_variables###
    lstm_variables_list = []
    weights_variables_list = [W_conv1a,W_conv2a,W_conv3a,W_conv4a,W_att1a,W_att2a,W_conv1s,W_conv2s,W_conv3s,W_conv4s,W_att1s,W_att2s,W11,W1]
    baias_variables_list = [b_conv1a,b_conv2a,b_conv3a,b_conv4a,b_conv1s,b_conv2s,b_conv3s,b_conv4s,b11,b1]
    all_variables_list = [lstm_variables_list,weights_variables_list,baias_variables_list]


    ##tensorboard
    tf.summary.histogram('W11_weights(feature)', W11)
    tf.summary.histogram('b11_weights(feature)', b11)
    tf.summary.histogram('W1_weights(classpre)', W1)
    tf.summary.histogram('b1_weights(classpre)', b1)
    tf.summary.histogram('W_conv1a_weights(conv)', W_conv1a)
    tf.summary.histogram('W_conv2a_weights(conv)', W_conv2a)
    tf.summary.histogram('W_conv1s_weights(conv)', W_conv1s)
    tf.summary.histogram('W_conv2s_weights(conv)', W_conv2s)
    summary = tf.summary.merge_all()

    return summary, all_variables_list

def train_and_evaluate(sess,sess_tag,summary,model_save_param,grad_scale=None, num_batches=None,input_data=None,filenames=None,domain_name_list=None):

    #input dataset
    [X_a,y_a,X_aT, y_aT,X_b,y_b,X_bT, y_bT] = input_data
    [X_filename,XT_filename] = filenames

    # Create batch builders
    A_batches = batch_generator([X_a, y_a], batch_size // 2)
    B_batches = batch_generator([X_b, y_b], batch_size // 2)

    # Get output tensors and train op
    d_at_acc = sess.graph.get_tensor_by_name('d_at_acc:0')
    d_xm_acc = sess.graph.get_tensor_by_name('d_xm_acc:0')
    p_acc = sess.graph.get_tensor_by_name('p_acc:0')
    eval_acc = sess.graph.get_tensor_by_name('eval_acc:0')
    domain_loss = sess.graph.get_tensor_by_name('domain_loss:0')
    domain_loss_atspeed = sess.graph.get_tensor_by_name('domain_loss_atspeed:0')
    domain_loss_xmean = sess.graph.get_tensor_by_name('domain_loss_xmean:0')
    pred_loss = sess.graph.get_tensor_by_name('pred_loss:0')

    saver = tf.train.Saver(max_to_keep=1000)###for model save

    sess.run(tf.global_variables_initializer())##
    output_filename = datetime.now().strftime('%Y%m%d%H%M%S')+"("+sess_tag+")"
    ##output modeldata
    if os.path.exists("model/"+output_filename)==False:
        os.mkdir("model/"+output_filename)
    shutil.copyfile(os.path.basename(__file__),"model/"+output_filename+"/"+os.path.basename(__file__)+".modelsave")
    if os.path.exists("model_code/"+output_filename)==False:
        os.mkdir("model_code/"+output_filename)
    shutil.copyfile(os.path.basename(__file__),"model_code/"+output_filename+"/"+os.path.basename(__file__))

    ##tensorboard
    tf_writer = tf.summary.FileWriter("tensorboard/"+output_filename, sess.graph)
    print("Use this code for tensorboard: tensorboard --logdir='./tensorboard/"+output_filename+"'")

    ##prepare for csv
    flout  = open("accuracy/"+output_filename+".csv","w")
    writer = csv.writer(flout,lineterminator="\n")
    writer.writerows([["train_num","lp",domain_name_list[0]+":Dacc(at)",domain_name_list[0]+":Dacc(xm)",domain_name_list[0]+":Cacc",domain_name_list[1]+":Dacc(at)",domain_name_list[1]+":Dacc(xm)",domain_name_list[1]+":Cacc",domain_name_list[0]+":DaccT(at)",domain_name_list[0]+":DaccT(xm)",domain_name_list[0]+":CaccT",domain_name_list[1]+":DaccT(at)",domain_name_list[1]+":DaccT(xm)",domain_name_list[1]+":CaccT","Dacc(at)","Dacc(xm)","DaccT(at)","DaccT(xm)"]])
    flout.close()

    flout  = open("error_collection/"+output_filename+"_train.csv","w")
    writer = csv.writer(flout,lineterminator="\n")
    writer.writerows(np.array([X_filename]).T)
    flout.close()

    flout  = open("error_collection/"+output_filename+"_test.csv","w")
    writer = csv.writer(flout,lineterminator="\n")
    writer.writerows(np.array([XT_filename]).T)
    flout.close()

    for i in range(num_batches):

        # If no grad_scale, use a schedule
        if grad_scale is None:
            p = float(i) / num_batches
            lp = 2. / (1. + 1.4**(-10. * p)) - 1
        elif 0<grad_scale:
            if float(i)<=(num_batches/10):
                lp = 0
            elif (num_batches/10)<=float(i) and float(i)<=(num_batches/1.2):
                p = float(i-num_batches/10) / (num_batches/1.2-num_batches/10)
                lp = (grad_scale+1.) / (1.+grad_scale*(1.4**(-10. * p)) ) - 1.
            else:
                lp = grad_scale
        else:
            lp = grad_scale

        X0, y0 = next(A_batches)
        X1, y1 = next(B_batches)
        Xbatch = np.vstack([X0, X1])
        ybatch = np.hstack([y0, y1])
        D_labels = np.hstack([np.zeros(batch_size // 2, dtype=np.int32),
                              np.ones(batch_size // 2, dtype=np.int32)])

        ##学習率やプロセスはここで設定
        if 0<=i%18 and i%18<8:
            train_optimizer = "pred_at_train_op"
            _, pa,w_summary = sess.run([train_optimizer, p_acc,summary],
                               feed_dict={'X:0': Xbatch, 'Y_ind:0': ybatch, 'D_ind:0': D_labels,'l:0': 0, 'keep_prob:0':0.5})##False->True
        elif 8<=i%18 and i%18<16:
            train_optimizer = "domain_only_train_op_xmean"
            _, pa,w_summary = sess.run([train_optimizer, p_acc,summary],
                               feed_dict={'X:0': Xbatch, 'Y_ind:0': ybatch, 'D_ind:0': D_labels,'l:0': lp, 'keep_prob:0':0.5})##False->True
        else:
            train_optimizer = "clfeature_domain_lp_op"
            _, pa,w_summary = sess.run([train_optimizer, p_acc,summary],
                    feed_dict={'X:0': Xbatch, 'Y_ind:0': ybatch, 'D_ind:0': D_labels,'l:0' : lp/2.0,'keep_prob:0':0.5})##False->True#ここ書き換えてます！！！
        ##

        if i%50==0:
            tf_writer.add_summary(w_summary, i)##tensorboard

            dloss_atA,dloss_xmA,plossA,d_at_acc_A,d_xm_acc_A, pa_A,train_evalA = sess.run([domain_loss_atspeed,domain_loss_xmean,pred_loss,d_at_acc,d_xm_acc, p_acc,eval_acc], feed_dict={'X:0': X_a, 'Y_ind:0': y_a,
                        'D_ind:0': np.zeros(X_a.shape[0], dtype=np.int32), 'l:0': 1.0, 'keep_prob:0':1.0})
            dloss_atB,dloss_xmB,plossB,d_at_acc_B,d_xm_acc_B, pa_B,train_evalB = sess.run([domain_loss_atspeed,domain_loss_xmean,pred_loss,d_at_acc,d_xm_acc, p_acc,eval_acc], feed_dict={'X:0': X_b, 'Y_ind:0': y_b,
                        'D_ind:0': np.ones(X_b.shape[0], dtype=np.int32), 'l:0': 1.0, 'keep_prob:0':1.0})

            print("lp:"+str(lp))
            print("**Train_acc("+str(i)+")**")
            print("Adomain(at):"+str(d_at_acc_A)+"/Adomain(xm):"+str(d_xm_acc_A)+"/Aclass:"+str(pa_A)+"/ Bdomain(at):"+str(d_at_acc_B)+"/Bdomain(xm):"+str(d_xm_acc_B)+"/Bclass:"+str(pa_B)+"/dloss(at):"+str(np.mean([dloss_atA,dloss_atB]))+"/dloss(xm):"+str(np.mean([dloss_xmA,dloss_xmB]))+"/ploss:"+str(np.mean([plossA,plossB])))

            dloss_atAT,dloss_xmAT,plossAT,d_at_acc_AT,d_xm_acc_AT, pa_AT,test_evalA = sess.run([domain_loss_atspeed,domain_loss_xmean,pred_loss,d_at_acc,d_xm_acc, p_acc,eval_acc], feed_dict={'X:0': X_aT, 'Y_ind:0': y_aT,
                        'D_ind:0': np.zeros(X_aT.shape[0], dtype=np.int32), 'l:0': 1.0, 'keep_prob:0':1.0})
            dloss_atBT,dloss_xmBT,plossBT,d_at_acc_BT,d_xm_acc_BT, pa_BT,test_evalB = sess.run([domain_loss_atspeed,domain_loss_xmean,pred_loss,d_at_acc,d_xm_acc, p_acc,eval_acc], feed_dict={'X:0': X_bT, 'Y_ind:0': y_bT,
                        'D_ind:0': np.ones(X_bT.shape[0], dtype=np.int32), 'l:0': 1.0, 'keep_prob:0':1.0})

            print("**Test_acc("+str(i)+")**")
            print("Adomain(at):"+str(d_at_acc_AT)+"/Adomain(xm):"+str(d_xm_acc_AT)+"/Aclass:"+str(pa_AT)+"/ Bdomain(at):"+str(d_at_acc_BT)+"/Bdomain(xm):"+str(d_xm_acc_BT)+"/Bclass:"+str(pa_BT)+"/dloss(at):"+str(np.mean([dloss_atAT,dloss_atBT]))+"/dloss(xm):"+str(np.mean([dloss_xmAT,dloss_xmBT]))+"/ploss:"+str(np.mean([plossAT,plossBT])))

            ###result output!!!###
            #csv yomikomi
            flin=open("accuracy/"+output_filename+".csv","r")
            dataReader = csv.reader(flin)
            csv_data = []
            for line in dataReader:
                csv_data.append(line)
            flin.close()
            csv_data.append([i,lp,d_at_acc_A,d_xm_acc_A,pa_A,d_at_acc_B,d_xm_acc_B,pa_B,d_at_acc_AT,d_xm_acc_AT,pa_AT,d_at_acc_BT,d_xm_acc_BT,pa_BT,np.mean([d_at_acc_A,d_at_acc_B]),np.mean([d_xm_acc_A,d_xm_acc_B]),np.mean([d_at_acc_AT,d_at_acc_BT]),np.mean([d_xm_acc_AT,d_xm_acc_BT])])
            #csv kakikomi
            flout  = open("accuracy/"+output_filename+".csv","w")
            writer = csv.writer(flout,lineterminator="\n")
            writer.writerows(csv_data)
            flout.close()

            ##train error_collection
            csv_data = ikeda_lib.read_csv("error_collection/"+output_filename+"_train.csv")
            csv_data = np.hstack((csv_data,np.array([np.hstack((train_evalA,train_evalB))]).T))
            ikeda_lib.write_csv("error_collection/"+output_filename+"_train.csv",csv_data)

            ##train error_collection
            csv_data = ikeda_lib.read_csv("error_collection/"+output_filename+"_test.csv")
            csv_data = np.hstack((csv_data,np.array([np.hstack((test_evalA,test_evalB))]).T))
            ikeda_lib.write_csv("error_collection/"+output_filename+"_test.csv",csv_data)

            ##model save
            if i>=0:
                if i % (50*model_save_param) == 0:
                    saver.save(sess, "model/"+output_filename+"/step", global_step=i)

def main_fnc(train_num,grad_scale_para,dataset_dir_name,dA_cA,dA_cB,dB_cA,dB_cB,dA_name,dB_name):

    print("Please name this session:") # print("タグ名:")
    sess_tag = input()

    """print("同条件での学習回数:")
    train_num = int(input())

    print("最大勾配反転率(0~1):")
    grad_scale_para = int(input())"""

    print("# training epochs (e.g., 30000):") #print("学習epoch数:")
    train_batch_num = int(input())

    print("Store model params (each 50*n epoch; e.g., 40):") #print("モデルの保存(50*n epochごと):")
    model_save_param = int(input())

    
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./accuracy', exist_ok=True)
    os.makedirs('./model_code', exist_ok=True)
    os.makedirs('./tensorboard', exist_ok=True)
    os.makedirs('./error_collection', exist_ok=True)
    os.makedirs('./analysis_set', exist_ok=True)


    """print("domainA/classA:")
    dA_cA = input()
    print("domainA/classB:")
    dA_cB = input()
    print("domainB/classA:")
    dB_cA = input()
    print("domainB/classB:")
    dB_cB = input()

    print("domainA_name:")
    dA_name = input()
    print("domainB_name:")
    dB_name = input()"""

    """print("dataset_dir_name:")
    dataset_dir_name = input()"""

    ##domainA
    X_a,y_a,X_aT, y_aT,X_a_filename,X_aT_filename = get_train_and_test_datas([dataset_dir_name+"/"+dA_cA,dataset_dir_name+"/"+dA_cB],0.2)#テストに用いるデータは２割
    ##domainB
    X_b,y_b,X_bT, y_bT,X_b_filename,X_bT_filename = get_train_and_test_datas([dataset_dir_name+"/"+dB_cA,dataset_dir_name+"/"+dB_cB],0.2)

    tf.reset_default_graph()
    summary,_ = build_model()
    sess = tf.InteractiveSession()

    for session_num in range(train_num):
        print("session_num:"+str(session_num))
        sess_tag_name = sess_tag+"_sessnum"+str(session_num)

        train_and_evaluate(sess,sess_tag_name,summary,model_save_param,grad_scale=grad_scale_para, num_batches=train_batch_num,input_data=[X_a,y_a,X_aT, y_aT,X_b,y_b,X_bT, y_bT],filenames=[np.hstack((X_a_filename,X_b_filename)),np.hstack((X_aT_filename,X_bT_filename))],domain_name_list=[dA_name,dB_name])

    return



if __name__ == '__main__':

    ###get params from commandline###
    args = sys.argv
    if len(args)!=10:
        print("Usage:twodomain_train.py 'train_num' 'grad_scale_para' 'dataset_dir_name' 'dA_name' 'dA_cA' 'dA_cB' 'dB_name' 'dB_cA' 'dB_cB'")
        sys.exit()
    train_num = int(args[1])
    grad_scale_para = float(args[2])
    dataset_dir_name = args[3]
    dA_name = args[4]
    dA_cA = args[5]
    dA_cB = args[6]
    dB_name = args[7]
    dB_cA = args[8]
    dB_cB = args[9]
    ######

    main_fnc(train_num,grad_scale_para,dataset_dir_name,dA_cA,dA_cB,dB_cA,dB_cB,dA_name,dB_name)
