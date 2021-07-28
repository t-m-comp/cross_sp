import numpy as np
import math


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from time_feature_func_lib import *

from scipy import signal
import copy

def MaxMinaveragew10(trajectory_datas):#senchu_kokunusu

	##setting##
	w_size = 0
	##

	##features##
	##

	##prepare##
	plt.close()
	##

	##(ii)feature
	#setting
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	acceleration = calc_diff(speed_data,10)
	moving_feature2 = calc_values(trajectory_datas,'speed_acc',10,'average')
	#max_mavrsp = np.max(speed_data)
	#min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(moving_feature2)-w_size):
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])#これでいいか？？？
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature2
	speed_list = calc_values(trajectory_datas,"speed",10,"average")
	relmins = signal.argrelmin(speed_list, order=10)[0]
	relmaxs = signal.argrelmax(speed_list, order=10)[0]
	relmin_index = np.array([ (i in relmins) for i in range(len(speed_list)) ])
	relmax_index = np.array([ (i in relmaxs) for i in range(len(speed_list)) ])
	relmax_del_index = copy.copy(relmax_index)
	for i in range(len(relmax_del_index)):
	    if relmax_del_index[i]==1:
	        #maxのすぐあとにminが来るかどうか
	        for j in range(i+1,len(relmax_del_index)):
	            if relmin_index[j]==1:
	                last_flag = 0
	                break
	            elif relmax_del_index[j]==1:
	                last_flag = 0
	                relmax_del_index[i]=0
	                break
	            last_flag = 1
	        if last_flag==1:
	            relmax_del_index[i]=0
	            break
	relrange_index = relmin_index*(-1) + relmax_del_index
	###


	#for threthold
	feature_list_acc = []
	dec_flag = 0
	for i in range(len(relrange_index)-w_size):
		thr_feature = relrange_index[int((i*2+w_size)/2)]
		if dec_flag==1:
			if thr_feature==-1:
				one_feature_acc += speed_data[i]#min
				feature_list_acc.append(one_feature_acc/float(i-max_index))
				dec_flag = 0
		else:
			if thr_feature==1:
				one_feature_acc = -speed_data[i]#max
				max_index = i
				dec_flag = 1
	##

	return np.mean(feature_list_acc)

def AvrAccw10_curveW10_o30_pre3(trajectory_datas):#senchu_kokunusu

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	moving_feature2 = calc_values(trajectory_datas,'speed_acc',5,'average')
	max_mavrsp = np.max(speed_data)
	min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])#これでいいか？？？
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	curves = calc_curves(trajectory_datas)
	curves = calc_moving_values(curves,5,"average")
	relmins = signal.argrelmin(curves, order=30)[0]
	relmin_index = [ (i in relmins) for i in range(len(curves)) ]
	x_mae = 3
	relmin_index_2 = relmin_index[x_mae:]+[0]*x_mae

	#for threthold
	"""_buf = []
	for i in range(len(curves)-w_size):
		_buf.append(curves[int((i*2+w_size)/2)])
	K = 27
	thr = np.sort(_buf)[K]#[::-1]"""
	#plot
	for i in range(len(relmin_index_2)-w_size):
		thr_feature = relmin_index_2[int((i*2+w_size)/2)]
		if thr_feature>=0.5:#1:極小,0:それ以外
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	#plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	#plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="curves")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def Averagecheck(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	return np.mean(speed_data)

def MaxMincheck(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	return np.max(speed_data)-np.min(speed_data)

def MaxMinTime(trajectory_datas):#senchu_kokunusu

	##setting##
	w_size = 0
	##

	##features##
	##

	##prepare##
	plt.close()
	##

	##(ii)feature
	#setting
	speed_data = calc_values(trajectory_datas,'speed',10,'average')
	acceleration = calc_diff(speed_data,10)
	moving_feature2 = calc_values(trajectory_datas,'speed_acc',10,'average')
	#max_mavrsp = np.max(speed_data)
	#min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(moving_feature2)-w_size):
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])#これでいいか？？？
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature2
	speed_list = calc_values(trajectory_datas,"speed",0,"average")
	relmins = signal.argrelmin(speed_list, order=10)[0]
	relmaxs = signal.argrelmax(speed_list, order=10)[0]
	relmin_index = np.array([ (i in relmins) for i in range(len(speed_list)) ])
	relmax_index = np.array([ (i in relmaxs) for i in range(len(speed_list)) ])
	relmax_del_index = copy.copy(relmax_index)
	for i in range(len(relmax_del_index)):
	    if relmax_del_index[i]==1:
	        #maxのすぐあとにminが来るかどうか
	        for j in range(i+1,len(relmax_del_index)):
	            if relmin_index[j]==1:
	                last_flag = 0
	                break
	            elif relmax_del_index[j]==1:
	                last_flag = 0
	                relmax_del_index[i]=0
	                break
	            last_flag = 1
	        if last_flag==1:
	            relmax_del_index[i]=0
	            break
	relrange_index = relmin_index*(-1) + relmax_del_index
	###


	#for threthold
	feature_list_time = []
	dec_flag = 0
	for i in range(len(relrange_index)-w_size):
		thr_feature = relrange_index[int((i*2+w_size)/2)]
		if dec_flag==1:
			if thr_feature==-1:
				feature_list_time.append(i-max_index)
				dec_flag = 0
		else:
			if thr_feature==1:
				max_index = i
				dec_flag = 1
	##

	return np.mean(feature_list_time)

def MaxMinaveragew0(trajectory_datas):#senchu_kokunusu

	##setting##
	w_size = 0
	##

	##features##
	##

	##prepare##
	plt.close()
	##

	##(ii)feature
	#setting
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	acceleration = calc_diff(speed_data,10)
	moving_feature2 = calc_values(trajectory_datas,'speed_acc',10,'average')
	#max_mavrsp = np.max(speed_data)
	#min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(moving_feature2)-w_size):
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])#これでいいか？？？
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature2
	speed_list = calc_values(trajectory_datas,"speed",0,"average")
	relmins = signal.argrelmin(speed_list, order=10)[0]
	relmaxs = signal.argrelmax(speed_list, order=10)[0]
	relmin_index = np.array([ (i in relmins) for i in range(len(speed_list)) ])
	relmax_index = np.array([ (i in relmaxs) for i in range(len(speed_list)) ])
	relmax_del_index = copy.copy(relmax_index)
	for i in range(len(relmax_del_index)):
	    if relmax_del_index[i]==1:
	        #maxのすぐあとにminが来るかどうか
	        for j in range(i+1,len(relmax_del_index)):
	            if relmin_index[j]==1:
	                last_flag = 0
	                break
	            elif relmax_del_index[j]==1:
	                last_flag = 0
	                relmax_del_index[i]=0
	                break
	            last_flag = 1
	        if last_flag==1:
	            relmax_del_index[i]=0
	            break
	relrange_index = relmin_index*(-1) + relmax_del_index
	###


	#for threthold
	feature_list_acc = []
	dec_flag = 0
	for i in range(len(relrange_index)-w_size):
		thr_feature = relrange_index[int((i*2+w_size)/2)]
		if dec_flag==1:
			if thr_feature==-1:
				one_feature_acc += speed_data[i]#min
				feature_list_acc.append(one_feature_acc/float(i-max_index))
				dec_flag = 0
		else:
			if thr_feature==1:
				one_feature_acc = -speed_data[i]#max
				max_index = i
				dec_flag = 1
	##

	return np.mean(feature_list_acc)

def Accw20_MaxMin(trajectory_datas):#senchu_kokunusu

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	speed_data = calc_values(trajectory_datas,'speed',5,'average')
	acceleration = calc_diff(speed_data,10)
	moving_feature2 = calc_values(trajectory_datas,'speed_acc',10,'average')
	#max_mavrsp = np.max(speed_data)
	#min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(moving_feature2)-w_size):
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])#これでいいか？？？
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature1
	curves = calc_curves(trajectory_datas)
	curves = calc_moving_values(curves,5,"average")
	relmins = signal.argrelmin(curves, order=1)[0]
	relmin_index = [ (i in relmins) for i in range(len(curves)) ]
	x_mae = 1
	relmin_index_2 = relmin_index[x_mae:]+[0]*x_mae

	##feature2
	speed_list = calc_values(trajectory_datas,"speed",0,"average")
	relmins = signal.argrelmin(speed_list, order=10)[0]
	relmaxs = signal.argrelmax(speed_list, order=10)[0]
	relmin_index = np.array([ (i in relmins) for i in range(len(speed_list)) ])
	relmax_index = np.array([ (i in relmaxs) for i in range(len(speed_list)) ])
	relmax_del_index = copy.copy(relmax_index)
	for i in range(len(relmax_del_index)):
	    if relmax_del_index[i]==1:
	        #maxのすぐあとにminが来るかどうか
	        for j in range(i+1,len(relmax_del_index)):
	            if relmin_index[j]==1:
	                last_flag = 0
	                break
	            elif relmax_del_index[j]==1:
	                last_flag = 0
	                relmax_del_index[i]=0
	                break
	            last_flag = 1
	        if last_flag==1:
	            relmax_del_index[i]=0
	            break

	rel_dec_index = []
	dec_flag = 0
	for i in range(len(relmin_index)):
	    if dec_flag==1:
	        rel_dec_index.append(1)
	        if relmin_index[i]==1:
	            dec_flag = 0
	    else:
	        if relmax_del_index[i]!=1:
	            rel_dec_index.append(0)
	        else:
	            dec_flag = 1
	            rel_dec_index.append(1)
	###


	#for threthold
	"""_buf = []
	for i in range(len(curves)-w_size):
		_buf.append(curves[int((i*2+w_size)/2)])
	K = 27
	thr = np.sort(_buf)[K]#[::-1]"""
	#plot
	for i in range(len(rel_dec_index)-w_size):
		thr_feature = rel_dec_index[int((i*2+w_size)/2)]
		if thr_feature>=0.5:#1:極小,0:それ以外
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	#plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	#plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="curves")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))


def MinSpeedw100_avrspeed50worst5per(trajectory_datas):#KOKUNUSUTO!!!

	##function##

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_feature2 = calc_values(trajectory_datas,'speed',50,'min')
	max_mavrsp = np.max(speed_data)
	min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_WeightAvr(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#F_abs_sum = np.sum(F_abs_range[0]:F_abs_range[1]);fft_ylist.append( (0 if F_abs_sum==0 else F_abs[1]/F_abs_sum ) )
		fft_ylist.append(moving_feature2[int((i+i+w_size)/2)])
		#fft_ylist.append((moving_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	moving_averagespeed = calc_values(trajectory_datas,'speed',25,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_averagespeed)-w_size):
		_buf.append(moving_averagespeed[int((i*2+w_size)/2)])
	K = 27
	thr = np.sort(_buf)[K]#[::-1]
	#plot
	for i in range(len(moving_averagespeed)-w_size):
		thr_feature = moving_averagespeed[int((i*2+w_size)/2)]
		if thr_feature<=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def MinSpeedw10_avrspeed20top20per(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_waidospeed = calc_values(trajectory_datas,'speed',10,'waido')
	max_sp = np.max(speed_data)
	min_sp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(moving_minspeed[int((i+i+w_size)/2)])
		fft_ylist.append((moving_minspeed[int((i+i+w_size)/2)]-min_sp)/(max_sp-min_sp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_avrspeed)-w_size):
		_buf.append(moving_avrspeed[int((i*2+w_size)/2)])
	K = 108
	thr = np.sort(_buf)[::-1][K]#topK
	#plot
	for i in range(len(moving_avrspeed)-w_size):
		thr_feature = moving_avrspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def ReMinSpeedw10_avracc20top10per_paper(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	max_sp = np.max(speed_data)
	min_sp = np.min(speed_data)
	mean_sp = np.mean(speed_data)
	std_sp = np.sqrt(np.var(speed_data))
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append((moving_minspeed[int((i+i+w_size)/2)]-min_sp)/(max_sp-min_sp))
		#fft_ylist.append((moving_minspeed[int((i+i+w_size)/2)]-mean_sp)/std_sp)

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	"""F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)"""
	##
	##feature2
	#setting
	moving_avracc = calc_values(trajectory_datas,'speed_acc',10,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',50,'min')
	#for threthold
	_buf = []
	for i in range(len(moving_avracc)-w_size):
		_buf.append(moving_avracc[int((i*2+w_size)/2)])
	K = 55
	thr = np.sort(_buf)[::-1][K]
	"""_buf2 = []
	for i in range(len(moving_minspeed)-w_size):
		if thr
		_buf.append(moving_minspeed[int((i*2+w_size)/2)])
	K = 50
	thr = np.sort(_buf)[::-1][K]"""
	#plot
	for i in range(len(moving_avracc)-w_size):
		thr_feature = moving_avracc[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##
	##feature3
	#setting
	moving_avracc = calc_values(trajectory_datas,'speed_acc',20,'average')
	moving_avracc = calc_values(trajectory_datas,'speed_acc',50,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_avracc)-w_size):
		_buf.append(moving_avracc[int((i*2+w_size)/2)])
	K = 30
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_avracc)-w_size):
		thr_feature = moving_avracc[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def FFTmaxFreqTop1_w200_spthr25avrtop5per_paper(trajectory_datas):

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		return return_value/float(np.sum(freq_list))

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 1
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init


		return freq

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 200
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))

	fft_ylist = calc_moving_values(fft_ylist,0,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	#for threthold
	_buf = []
	for i in range(len(moving_avrspeed)-w_size):
		_buf.append(moving_avrspeed[int((i*2+w_size)/2)])
	K = 27
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_avrspeed)-w_size):
		thr_feature = moving_avrspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def ReAvrSpeedw0_minspeed20top5per_paper(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 100
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_avrspeed_feature2 = calc_values(trajectory_datas,'speed',0,'average')
	max_mavrsp = np.max(speed_data)
	min_mavrsp = np.min(speed_data)
	mean_mavrsp = np.mean(speed_data)
	std_mavrsp = np.sqrt(np.var(speed_data))
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_WeightAvr(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#F_abs_sum = np.sum(F_abs_range[0]:F_abs_range[1]);fft_ylist.append( (0 if F_abs_sum==0 else F_abs[1]/F_abs_sum ) )
		#fft_ylist.append(moving_avrspeed_feature2[int((i+i+w_size)/2)])
		fft_ylist.append((moving_avrspeed_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))
		#fft_ylist.append((moving_avrspeed_feature2[int((i+i+w_size)/2)]-mean_mavrsp)/std_mavrsp)

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	#for threthold
	_buf = []
	for i in range(len(moving_minspeed)-w_size):
		_buf.append(moving_minspeed[int((i*2+w_size)/2)])
	K = 27
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_minspeed)-w_size):
		thr_feature = moving_minspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))
#####fundamental feature#####

def average_speed(trajectory_datas):

	return np.mean(trajectory_datas[:,2])

def variance_speed(trajectory_datas):

	return np.var(trajectory_datas[:,2])

def angle_speed_average(trajectory_datas):

	angle_speed_list = trajectory_datas[:,1]

	return np.mean(np.abs(angle_speed_list))

def angle_speed_variance(trajectory_datas):

	angle_speed_list = trajectory_datas[:,1]

	return np.var(np.abs(angle_speed_list))

def angle_speed_variance2(trajectory_datas):

	angle_speed_list = trajectory_datas[:,1]

	return np.var(angle_speed_list)

def average_acc(trajectory_datas):

	speed = trajectory_datas[:,2]

	return np.mean(speed[1:]-speed[:-1])

def variance_acc(trajectory_datas):

	speed = trajectory_datas[:,2]

	return np.var(speed[1:]-speed[:-1])

def average_angle_acc(trajectory_datas):

	speed = trajectory_datas[:,1]

	return np.mean(speed[1:]-speed[:-1])

##spatial feature##
def FFT_histogramALL(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	F = np.fft.fft(speed_data)
	F_abs = np.abs(F)

	return F_abs[:int(len(F_abs)/2)][1:],538
def only_joken_speedkeep(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',1,'average')
	moving_check = calc_values(trajectory_datas,'speed',50,'average')#trajectory_datas[:,5]#
	thr_value = np.mean(moving_check)

	test = 0
	keep_time = 0
	keep_time_list = []
	for i in range(len(speed_data)):
		if moving_check[i]>=thr_value:
			keep_time += 1
		else:
			if keep_time>=10:
				keep_time_list.append(keep_time)
			keep_time = 0
		test = 1
	keep_time_list.append(keep_time)

	"""plt.plot(moving_check)
	plt.plot([thr_value+0.0]*len(moving_check))
	plt.show()"""

	"""if test!=1:
	print(thr_value)
	plt.plot(moving_check)
	plt.show()"""

	if len(keep_time_list)!=0:
		return np.mean(keep_time_list)
	else:
		return 0
def only_joken_min10(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',1,'average')
	moving_check = calc_values(trajectory_datas,'speed',10,'min')#trajectory_datas[:,5]#
	thr_value = np.mean(moving_check)

	test = 0
	average_value = []
	for i in range(len(speed_data)):
		if moving_check[i]>=thr_value:
			average_value.append(speed_data[i])
			test = 1

	"""plt.plot(moving_check)
	plt.plot([thr_value+0.0]*len(moving_check))
	plt.show()"""

	"""if test!=1:
	print(thr_value)
	plt.plot(moving_check)
	plt.show()"""

	return np.mean(average_value)

def speed_histogram_minmax(trajectory_datas):

	histogram_min = -8
	histogram_max = 8
	histogram_unit = 0.1
	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)

	speed_max = np.max(speed_data)
	speed_min = np.min(speed_data)
	for current_speed in np.arange(speed_min,speed_max, histogram_unit):
		if histogram_min<current_speed and current_speed<histogram_max:
			#print(current_speed)
			#print(int((current_speed-histogram_min)/histogram_unit)+1)
			#s = input()
			histogram_list[int((current_speed-histogram_min)/histogram_unit)+1] += 1
		elif current_speed<=histogram_min:
			histogram_list[0] += 1
		elif histogram_max<=current_speed:
			histogram_list[-1] += 1

	"""plt.plot(moving_checkspeed)
	plt.plot([thr_speed+0.0]*len(moving_checkspeed))
	plt.show()"""

	"""if test!=1:
		print(thr_speed)
		plt.plot(moving_checkspeed)
		plt.show()"""

	return np.array(histogram_list) / sum(histogram_list)

def speed_histogram_all(trajectory_datas):

	histogram_min = -1
	histogram_max = 1
	histogram_unit = 0.1
	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)):
		current_speed = speed_data[i]
		if histogram_min<current_speed and current_speed<histogram_max:
			#print(current_speed)
			#print(int((current_speed-histogram_min)/histogram_unit)+1)
			#s = input()
			histogram_list[int((current_speed-histogram_min)/histogram_unit)+1] += 1
		elif current_speed<=histogram_min:
			histogram_list[0] += 1
		elif histogram_max<=current_speed:
			histogram_list[-1] += 1

	"""plt.plot(moving_checkspeed)
	plt.plot([thr_speed+0.0]*len(moving_checkspeed))
	plt.show()"""

	"""if test!=1:
		print(thr_speed)
		plt.plot(moving_checkspeed)
		plt.show()"""

	return np.array(histogram_list) / sum(histogram_list)

def FFT_histogram_joken(trajectory_datas):

	w_size = 100
	F_abs_sum = np.array([0.0]*w_size)
	speed_data = calc_values(trajectory_datas,'speed',1,'average')
	moving_check = trajectory_datas[:,5]#calc_values(trajectory_datas,'speed',50,'min')
	thr_value = np.mean(moving_check)

	test = 0
	for i in range(len(speed_data)-w_size):
		if moving_check[int((i*2+w_size)/2)]>=thr_value+0.0:
			F = np.fft.fft(speed_data[i:i+w_size])
			F_abs = np.abs(F)
			#normalize
			F_abs /= sum(F_abs)
			F_abs_sum += F_abs
			test = 1

	"""plt.plot(moving_check)
	plt.plot([thr_value+0.0]*len(moving_check))
	plt.show()"""

	"""if test!=1:
	print(thr_value)
	plt.plot(moving_check)
	plt.show()"""

	F_abs_output = F_abs_sum[:int(len(F_abs_sum)/2)][1:]

	if sum(F_abs_output)!=0:
		return F_abs_output/sum(F_abs_output),w_size
	else:
		return F_abs_output,w_size

def speed_histogram(trajectory_datas):

	w_size = 100
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	speed_data = calc_values(trajectory_datas,'speed',50,'min')
	moving_checkspeed = calc_values(trajectory_datas,'speed',0,'average')
	thr_speed = np.max(moving_checkspeed)#np.mean(moving_checkspeed)

	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		if moving_checkspeed[int((i*2+w_size)/2)]>=thr_speed+0.0:
			for j in range(w_size):
				current_speed = speed_data[i+j]
				if histogram_min<current_speed and current_speed<histogram_max:
					histogram_list[int((current_speed-histogram_min)/histogram_unit)+1] += 1
				elif current_speed<=histogram_min:
					histogram_list[0] += 1
				elif histogram_max<=current_speed:
					histogram_list[-1] += 1

	"""plt.plot(moving_checkspeed)
	plt.plot([thr_speed+0.0]*len(moving_checkspeed))
	plt.show()"""

	"""if test!=1:
		print(thr_speed)
		plt.plot(moving_checkspeed)
		plt.show()"""

	if sum(histogram_list)!=0:
		histogram_list = (np.array(histogram_list) / sum(histogram_list)).tolist()

	return histogram_list,w_size
def minspeed_check(trajectory_datas):

	speed_data = calc_values(trajectory_datas,'speed',30,'average')
	moving_checkspeed = calc_values(trajectory_datas,'speed',1,'average')
	thr_speed = np.max(moving_checkspeed)#np.mean(moving_checkspeed)

	minspeed = 0
	count = 0
	for i in range(len(speed_data)):
		if moving_checkspeed[i]>=thr_speed:
			minspeed += speed_data[i]
			count += 1

	return minspeed/float(count)

def FFT_histogram(trajectory_datas):

	#w_size = 537
	speed_data = calc_values(trajectory_datas,'speed',0,'average')

	F = np.fft.fft(speed_data)
	F_abs = np.abs(F)

	return F_abs[:int(len(F_abs)/2)][1:]

def FFTlow_minspeed(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	thr = 1100
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'min')

	#plt.plot(moving_avrspeed)
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs)>=thr:
			#plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
			current_feature = moving_avrspeed[int((i*2+w_size)/2)]
			if histogram_min<current_feature and current_feature<histogram_max:
				histogram_list[int((current_feature-histogram_min)/histogram_unit)+1] += 1
			elif current_feature<=histogram_min:
				histogram_list[0] += 1
			elif histogram_max<=current_feature:
				histogram_list[-1] += 1
	#plt.show()

	return histogram_list[:]#,w_size#int(len(F_abs_sum)/3)+3

def FFTlowmax_speed(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	#plt.plot(moving_avrspeed)
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs))
	thr = np.max(_buf)
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs)>=thr:
			#plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
			current_feature = moving_avrspeed[int((i*2+w_size)/2)]
			if histogram_min<current_feature and current_feature<histogram_max:
				histogram_list[int((current_feature-histogram_min)/histogram_unit)+1] += 1
			elif current_feature<=histogram_min:
				histogram_list[0] += 1
			elif histogram_max<=current_feature:
				histogram_list[-1] += 1
	#plt.show()

	return histogram_list[:]#,w_size#int(len(F_abs_sum)/3)+3

def FFTsumhigh_speed(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	#thr = 1200
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'min')
	#moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[:int(w_size/10)]))
	thr = np.max(_buf)*0.9

	#plt.plot(speed_data)
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs[:int(w_size/10)])>=thr:
			#plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
			current_feature = moving_avrspeed[int((i*2+w_size)/2)]
			if histogram_min<current_feature and current_feature<histogram_max:
				histogram_list[int((current_feature-histogram_min)/histogram_unit)+1] += 1
			elif current_feature<=histogram_min:
				histogram_list[0] += 1
			elif histogram_max<=current_feature:
				histogram_list[-1] += 1
			#plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
	#plt.show()

	return histogram_list[:]

def FFTsumhigh_speedminavr(trajectory_datas):

	w_size = 100
	#thr = 1200
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'min')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[:]))#int(w_size/10)
	thr = np.max(_buf)*0.8

	plt.close()
	plt.plot(moving_avrspeed)
	feature_list = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs)>=thr:
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
			current_feature = moving_avrspeed[int((i*2+w_size)/2)]
			feature_list.append(current_feature)
	plt.savefig("_/"+str(speed_data[100])+".png")
	plt.close()

	return np.mean(feature_list)#,w_size#int(len(F_abs_sum)/3)+3

def FFTattention_check_K150_portion(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	#F_abs_range = (0,int(w_size/10))
	F_abs_range = (2,int(w_size/10))
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 150
	thr = np.sort(_buf)[::-1][K]
	#thr = 1200

	plt.close()
	thr_feature_xlist = []
	thr_feature_ylist = []
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature_xlist.append(int((i+i+w_size)/2))
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		thr_feature_ylist.append(thr_feature)
		if thr_feature>=thr:
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
	plt.plot(thr_feature_xlist,calc_moving_values(thr_feature_ylist,40,"average"))
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return histogram_list[:]#,w_size#int(len(F_abs_sum)/3)+3

def FFTattention_check_K150_elim1_mavr(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	#F_abs_range = (0,int(w_size/10))
	F_abs_range = (2,int(w_size/10))
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 150
	thr = np.sort(_buf)[::-1][K]
	#thr = 1200

	plt.close()
	thr_feature_xlist = []
	thr_feature_ylist = []
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature_xlist.append(int((i+i+w_size)/2))
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		thr_feature_ylist.append(thr_feature)
		if thr_feature>=thr:
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
	plt.plot(thr_feature_xlist,calc_moving_values(thr_feature_ylist,40,"average"))
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return histogram_list[:]#,w_size#int(len(F_abs_sum)/3)+3


def AvrSpeedw10_avrspeed20top100(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	moving_waidospeed = calc_values(trajectory_datas,'speed',10,'waido')
	max_sp = np.max(speed_data)
	min_sp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(moving_minspeed[int((i+i+w_size)/2)])
		fft_ylist.append((moving_avrspeed[int((i+i+w_size)/2)]-min_sp)/(max_sp-min_sp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	"""F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)"""
	##
	##feature2
	#setting
	moving_avracc = calc_values(trajectory_datas,'speed',10,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	#for threthold
	_buf1 = []
	for i in range(len(moving_avracc)-w_size):
		_buf1.append(moving_avracc[int((i*2+w_size)/2)])
	K = 100
	thr1 = np.sort(_buf1)[::-1][K]#topK
	_buf2 = []
	for i in range(len(moving_minspeed)-w_size):
		if moving_avracc[int((i*2+w_size)/2)]>=thr1:
			_buf2.append(moving_minspeed[int((i*2+w_size)/2)])
	K = int(len(_buf2)/3)
	thr2 = np.sort(_buf2)[::-1][K]#topK
	#plot
	for i in range(len(moving_avracc)-w_size):
		thr_feature1 = moving_avracc[int((i*2+w_size)/2)]
		thr_feature2 = moving_minspeed[int((i*2+w_size)/2)]
		if thr_feature1>=thr1 and thr_feature2>=thr2:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##
	##feature3
	#setting
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_avrspeed)-w_size):
		_buf.append(moving_avrspeed[int((i*2+w_size)/2)])
	K = 100
	thr = np.sort(_buf)[::-1][K]#topK
	#plot
	for i in range(len(moving_avrspeed)-w_size):
		thr_feature = moving_avrspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def ReMinSpeedw10_avracc20minspeed50top100X20(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	max_sp = np.max(speed_data)
	min_sp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		fft_ylist.append((moving_minspeed[int((i+i+w_size)/2)]-min_sp)/(max_sp-min_sp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	"""F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)"""
	##
	##feature2
	#setting
	moving_avracc = calc_values(trajectory_datas,'speed_acc',10,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',25,'min')
	#for threthold
	_buf1 = []
	for i in range(len(moving_avracc)-w_size):
		_buf1.append(moving_avracc[int((i*2+w_size)/2)])
	K = 100
	thr1 = np.sort(_buf1)[::-1][K]#topK
	_buf2 = []
	for i in range(len(moving_minspeed)-w_size):
		if moving_avracc[int((i*2+w_size)/2)]>=thr1:
			_buf2.append(moving_minspeed[int((i*2+w_size)/2)])
	K = int(len(_buf2)/5)
	thr2 = np.sort(_buf2)[::-1][K]#topK
	#plot
	for i in range(len(moving_avracc)-w_size):
		thr_feature1 = moving_avracc[int((i*2+w_size)/2)]
		thr_feature2 = moving_minspeed[int((i*2+w_size)/2)]
		if thr_feature1>=thr1 and thr_feature2>=thr2:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##
	##feature3
	#setting
	moving_avracc = calc_values(trajectory_datas,'speed_acc',20,'average')
	moving_avracc = calc_values(trajectory_datas,'speed_acc',50,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_avracc)-w_size):
		_buf.append(moving_avracc[int((i*2+w_size)/2)])
	K = 30
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_avracc)-w_size):
		thr_feature = moving_avracc[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

###########################################################################

def MinSpeedw10_avrspeed20top20per(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',5,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_waidospeed = calc_values(trajectory_datas,'speed',10,'waido')
	max_sp = np.max(speed_data)
	min_sp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(moving_minspeed[int((i+i+w_size)/2)])
		fft_ylist.append((moving_minspeed[int((i+i+w_size)/2)]-min_sp)/(max_sp-min_sp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##
	##feature3
	#setting
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	#for threthold
	_buf = []
	for i in range(len(moving_avrspeed)-w_size):
		_buf.append(moving_avrspeed[int((i*2+w_size)/2)])
	K = 108
	thr = np.sort(_buf)[::-1][K]#topK
	#plot
	for i in range(len(moving_avrspeed)-w_size):
		thr_feature = moving_avrspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))


###############################################################################

def ReAvrSpeedw0_minspeed20top30(trajectory_datas):#maxFreqTop3#undercut

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_avrspeed_feature2 = calc_values(trajectory_datas,'speed',0,'average')
	max_mavrsp = np.max(speed_data)
	min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_WeightAvr(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#F_abs_sum = np.sum(F_abs_range[0]:F_abs_range[1]);fft_ylist.append( (0 if F_abs_sum==0 else F_abs[1]/F_abs_sum ) )
		#fft_ylist.append(moving_avrspeed_feature2[int((i+i+w_size)/2)])
		fft_ylist.append((moving_avrspeed_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##
	##feature2
	#setting
	F_abs_range = (2,int(w_size/2))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.004,-0.004], linewidth = 3.0,color="red")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##
	##feature3
	#setting
	#for threthold
	_buf = []
	for i in range(len(moving_minspeed)-w_size):
		_buf.append(moving_minspeed[int((i*2+w_size)/2)])
	K = 30
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_minspeed)-w_size):
		thr_feature = moving_minspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def ReWaidoSpeedw10_avrspeed50worst50(trajectory_datas):#KOKUNUSUTO!!!

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		if np.sum(freq_list)!=0:
			return return_value/float(np.sum(freq_list))
		else:
			return 0

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 3
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init

		return freq

	def calc_avrspeed(freq_list,first_init):

		return freq_list[0]

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 0
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	moving_avrspeed_feature2 = calc_values(trajectory_datas,'speed',25,'average')
	max_mavrsp = np.max(speed_data)
	min_mavrsp = np.min(speed_data)
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		#F = np.fft.fft(speed_data[i:i+w_size])
		#F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_WeightAvr(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#F_abs_sum = np.sum(F_abs_range[0]:F_abs_range[1]);fft_ylist.append( (0 if F_abs_sum==0 else F_abs[1]/F_abs_sum ) )
		#fft_ylist.append(moving_avrspeed_feature2[int((i+i+w_size)/2)])
		fft_ylist.append((moving_avrspeed_feature2[int((i+i+w_size)/2)]-min_mavrsp)/(max_mavrsp-min_mavrsp))

	#fft_ylist = calc_moving_values(fft_ylist,5,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	"""F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)"""
	##
	##feature2
	#setting
	"""F_abs_range = (2,int(w_size/2))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.004,-0.004], linewidth = 3.0,color="red")
		else:
			_ = 0
			#attention_calc_list.append(0)"""
	##
	##feature3
	#setting
	moving_waidospeed = calc_values(trajectory_datas,'speed',5,'waido')
	#for threthold
	_buf = []
	for i in range(len(moving_waidospeed)-w_size):
		_buf.append(moving_waidospeed[int((i*2+w_size)/2)])
	K = 50
	thr = np.sort(_buf)[K]#[::-1]
	#plot
	for i in range(len(moving_waidospeed)-w_size):
		thr_feature = moving_waidospeed[int((i*2+w_size)/2)]
		if thr_feature<=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def FFTmaxFreqTop1_w200_spthr25avrtop50(trajectory_datas):

	##function##
	def calc_WeightAvr(freq_list,first_init):

		return_value = 0
		for i in range(len(freq_list)):
			return_value += (i+first_init)*freq_list[i]

		return return_value/float(np.sum(freq_list))

	def calc_WeightAvr_premax(freq_list,first_init):

		max_index = np.argmax(freq_list)
		max_weight = 1

		return_value = 0
		for i in range(len(freq_list)):
			if i!=max_index:
				return_value += (i+first_init)*freq_list[i]
			else:
				return_value += (i+first_init)*freq_list[i]*max_weight

		return return_value/float(np.sum(freq_list)+freq_list[i]*(max_weight-1))

	def calc_maxFreq(freq_list,first_init):

		return np.argmax(freq_list)+first_init

	def calc_maxFreqTopK(freq_list,first_init):

		topK = 1
		thr = np.sort(freq_list)[::-1][topK-1]

		#freq = np.mean(freq_list[np.where(freq_list >= thr, True, False)])+first_init
		freq = np.mean(np.arange(0,len(freq_list))[np.where(freq_list >= thr, True, False)])+first_init


		return freq

	##setting##
	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1
	w_size = 200
	##

	##features##
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',25,'average')
	moving_minspeed = calc_values(trajectory_datas,'speed',10,'min')
	##

	##prepare##
	attention_calc_list = []
	plt.close()
	##

	##(ii)feature
	#setting
	F_abs_range = (1,int(w_size/10))
	#plot
	fft_xlist = []
	fft_ylist = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		fft_xlist.append(int((i+i+w_size)/2))
		#fft_ylist.append(calc_WeightAvr_premax(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		#fft_ylist.append(calc_maxFreq(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))
		fft_ylist.append(calc_maxFreqTopK(F_abs[F_abs_range[0]:F_abs_range[1]],F_abs_range[0])/float(w_size))

	fft_ylist = calc_moving_values(fft_ylist,0,"average")
	plt.plot(fft_xlist,fft_ylist)#,label="fft"
	##

	###(i)attention###
	##feature1
	#setting
	F_abs_range = (2,int(w_size/10))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.002,-0.002], linewidth = 3.0,color="green")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##
	##feature2
	#setting
	F_abs_range = (2,int(w_size/2))
	#for threthold
	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[F_abs_range[0]:F_abs_range[1]]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		thr_feature = np.sum(F_abs[F_abs_range[0]:F_abs_range[1]])
		if thr_feature>=thr:
			#attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.004,-0.004], linewidth = 3.0,color="red")
		else:
			_ = 0
			#attention_calc_list.append(0)
	##
	##feature3
	#setting
	#for threthold
	_buf = []
	for i in range(len(moving_avrspeed)-w_size):
		_buf.append(moving_avrspeed[int((i*2+w_size)/2)])
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#plot
	for i in range(len(moving_avrspeed)-w_size):
		thr_feature = moving_avrspeed[int((i*2+w_size)/2)]
		if thr_feature>=thr:
			attention_calc_list.append(1)
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[-0.006,-0.006], linewidth = 3.0,color="pink")
		else:
			_ = 0
			attention_calc_list.append(0)
	##

	##last plot
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="green",label="fftsum(~0.1)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="red",label="fftsum(all)")
	plt.plot([100,100],[0.01,0.01], linewidth = 2,color="pink",label="speed")
	plt.legend()
	plt.text(100,0.002,s="{0:.2f}".format(np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))))
	#plt.colorbar()
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return np.sum((np.array(fft_ylist)*np.array(attention_calc_list)))/float(np.sum(attention_calc_list))

def FFTsumhigh_speed(trajectory_datas):

	histogram_min = -3
	histogram_max = 3
	histogram_unit = 0.1

	w_size = 100
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[:int(w_size/10)]))#
	K = 50
	thr = np.sort(_buf)[::-1][K]
	#thr = 1200

	plt.close()
	plt.plot(moving_avrspeed)
	histogram_list = [0]*(int((histogram_max-histogram_min)/histogram_unit)+3)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs[:int(w_size/10)])>=thr:
			plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
			current_feature = moving_avrspeed[int((i*2+w_size)/2)]
			if histogram_min<current_feature and current_feature<histogram_max:
				histogram_list[int((current_feature-histogram_min)/histogram_unit)+1] += 1
			elif current_feature<=histogram_min:
				histogram_list[0] += 1
			elif histogram_max<=current_feature:
				histogram_list[-1] += 1
	#plt.savefig("_/"+str(speed_data[100])+".png")
	#plt.close()

	return histogram_list[:]#,w_size#int(len(F_abs_sum)/3)+3

def FFTsumhigh_FFT(trajectory_datas):

	histogram_unit = 0.1

	w_size = 100
	#thr = 1200
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	#moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')

	_buf = []
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		_buf.append(np.sum(F_abs[:int(w_size/10)]))
	thr = np.max(_buf)*0.9

	#plt.plot(speed_data)
	F_abs_sum = np.array([0.0]*w_size)
	for i in range(len(speed_data)-w_size):
		F = np.fft.fft(speed_data[i:i+w_size])
		F_abs = np.abs(F)
		if np.sum(F_abs[:int(w_size/10)])>=thr:
			F_abs_sum += F_abs
			#plt.plot([int((i*2+w_size)/2),int((i*2+w_size)/2)+1],[0,0],color="pink")
	#plt.show()

	return F_abs_sum[:int(len(F_abs)/2)][1:],w_size

def FFT_histogram100_max_upspeedNO(trajectory_datas):

	w_size = 100
	F_abs_sum = np.array([0.0]*(int(w_size/2)-1))
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	average_speed = np.mean(speed_data)

	"""plt.plot(moving_avrspeed)
	plt.plot([average_speed+0.3]*len(moving_avrspeed))
	plt.show()"""

	for i in range(len(speed_data)-w_size):
		if True:#moving_avrspeed[int((i*2+w_size)/2)]>=average_speed+0.3:
			F = np.fft.fft(speed_data[i:i+w_size])
			F_abs = np.abs(F)
			F_abs_sum[np.argmax(F_abs[:int(len(F_abs)/2)][1:])] += 1

	return F_abs_sum[:],w_size#int(len(F_abs_sum)/3)+3
def FFT_histogram100_max_upspeed1t4(trajectory_datas):

	w_size = 100
	F_abs_sum = np.array([0.0]*(int(w_size/2)-1))
	speed_data = calc_values(trajectory_datas,'speed',0,'average')
	moving_avrspeed = calc_values(trajectory_datas,'speed',10,'average')
	average_speed = np.mean(speed_data)

	plt.plot(moving_avrspeed)
	plt.plot([average_speed]*len(moving_avrspeed))
	plt.show()

	for i in range(len(speed_data)-w_size):
		if moving_avrspeed[int((i*2+w_size)/2)]>=average_speed*1.4:
			F = np.fft.fft(speed_data[i:i+w_size])
			F_abs = np.abs(F)
			F_abs_sum[np.argmax(F_abs[:int(len(F_abs)/2)][1:])] += 1

	return F_abs_sum[:],w_size#int(len(F_abs_sum)/3)+3

#####optional feature#####

def speed_variance100(trajectory_datas):

	return calc_values(trajectory_datas,'speed',50,'variance')

def speed_waido10(trajectory_datas):

	return calc_values(trajectory_datas,'speed',5,'waido')

def speed_average0(trajectory_datas):

	return calc_values(trajectory_datas,'speed',0,'average')

def speed_min10(trajectory_datas):

	return calc_values(trajectory_datas,'speed',10,'min')

def attention_check(attention_type,time_feature_discript,trajectory_datas):

	if attention_type=="speed":
		attention_list = trajectory_datas[:,5]
	elif attention_type=="angle":
		attention_list = trajectory_datas[:,6]

	tdata = trajectory_datas
	time_feature = eval(time_feature_discript)

	feature_xmean = 0
	for i in range(len(attention_list)):
		feature_xmean += time_feature[i] * attention_list[i]

	return feature_xmean

def attention_check_none(attention_type,time_feature_discript,trajectory_datas):

	"""if attention_type=="speed":
		attention_list = trajectory_datas[:,5]
	elif attention_type=="angle":
		attention_list = trajectory_datas[:,6]"""

	tdata = trajectory_datas
	time_feature = eval(time_feature_discript)

	feature_xmean = 0
	for i in range(len(time_feature)):
		feature_xmean += time_feature[i] / float(len(time_feature))

	return feature_xmean

##attention##

def speed_keep(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,5,"average")
	window_speed_list2 = calc_moving_values(speed_list,30,"average")

	speed_average = 0
	speed_count = 0
	for i in range(len(window_speed_list)):
		speed_wide = window_speed_list[i]
		if 1<=speed_wide:
			speed_average += window_speed_list2[i]
			speed_count += 1

	if speed_count!=0:
		return speed_average/float(speed_count)
	else:
		return 0

def speed_up_sharp(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,25,"average")
	speed_acc_list = time_feature_acc(trajectory_datas)
	window_speed_acc_list = calc_moving_values(speed_acc_list,40,"average")

	speed_average = 0
	speed_count = 0
	for i in range(len(window_speed_acc_list)):
		acc_wide = window_speed_acc_list[i]
		if 0.1<=acc_wide:
			speed_average += window_speed_list[i]
			speed_count += 1

	if speed_count!=0:
		return speed_average/float(speed_count)
	else:
		return 0

def speed_up_or_down(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,25,"average")
	speed_acc_list = time_feature_acc(trajectory_datas)

	acc_average = 0
	acc_count = 0
	for i in range(len(window_speed_list)):
		speed_wide = window_speed_list[i]
		if 1<=speed_wide and speed_wide<=2:
			acc_average += speed_acc_list[i]
			acc_count += 1

	if acc_count!=0:
		return acc_average/float(acc_count)
	else:
		return 0

def speed_down_ratio(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,25,"average")
	speed_acc_list = time_feature_acc(trajectory_datas)

	acc_down_count = 0
	acc_count = 0
	for i in range(len(window_speed_list)):
		speed_wide = window_speed_list[i]
		if 1<=speed_wide and speed_wide<=2:
			acc_count += 1
			if speed_acc_list[i]<0:
				acc_down_count += 1

	if acc_count!=0:
		return acc_down_count/float(acc_count)
	else:
		return 0

def speed_attention(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,25,"average")

	speed_count = 0
	for i in range(len(window_speed_list)):
		speed_wide = window_speed_list[i]
		if 1<=speed_wide and speed_wide<=2:
			speed_count += 1

	return speed_count

def angle_var2(trajectory_datas):

	angle_list = trajectory_datas[:,1]
	window_angle_avr_list = calc_moving_values(angle_list,50,"average")
	window_angle_var_list = calc_moving_values(angle_list,25,"variance")

	angle_var_avr = 0
	angle_plus_count = 0
	for i in range(len(window_angle_avr_list)):
		angle_avr = window_angle_avr_list[i]
		if -0.5<=angle_avr and angle_avr<=0.5:
			angle_var_avr += window_angle_var_list[i]
			angle_plus_count += 1

	if angle_plus_count!=0:
		return angle_var_avr/float(angle_plus_count)
	else:
		return None

def stop_ratio_angle(trajectory_datas):

	stop_angle_criteria = 0.05

	angle_list = trajectory_datas[:,1]
	window_angle_avr_list = calc_moving_values(angle_list,25,"average")
	count = 0
	for i in range(len(window_angle_avr_list)):
		if np.abs(window_angle_avr_list[i]) <= stop_angle_criteria:
			count = count+1

	return count/float(len(window_angle_avr_list))

def speed_avr_up(trajectory_datas):

	speed_list = trajectory_datas[:,2]
	window_speed_list = calc_moving_values(speed_list,25,"average")

	speed_avr = 0
	speed_plus_count = 0
	for i in range(len(window_speed_list)):
		if 0.1<=window_speed_list[i] and window_speed_list[i]<=0.3:
			speed_avr += window_speed_list[i]
			speed_plus_count += 1

	if speed_plus_count!=0:
		return speed_avr/float(speed_plus_count)
	else:
		return None

def speed_down_katamuki(trajectory_datas):

	speed_list = trajectory_datas[:,2]

	speed_list_2 = np.append(speed_list,speed_list[-1])
	speed_list_2 = np.delete(speed_list_2,0)

	acc_list = speed_list_2 - speed_list

	window_acc_list = calc_moving_values(acc_list,25,"average")

	acc_avr = 0
	acc_plus_count = 0
	for i in range(len(window_acc_list)):
		if window_acc_list[i]<=0.0:
			acc_avr += window_acc_list[i]
			acc_plus_count += 1

	if acc_plus_count!=0:
		return acc_avr/float(acc_plus_count)
	else:
		return None

##attention check routine##

def angle_attention_check(trajectory_datas):

	angle_list = trajectory_datas[:,1]
	angle_attention_list = trajectory_datas[:,6]
	window_angle_avr_list = calc_moving_values(angle_list,50,"average")
	window_angle_var_list = calc_moving_values(angle_list,50,"variance")

	angle_var_avr = 0
	angle_plus_count = 0
	for i in range(len(angle_attention_list)):
		angle_attention = angle_attention_list[i]
		if 0.001<=angle_attention:
			angle_var_avr += window_angle_var_list[i]
			angle_plus_count += 1

	if angle_plus_count!=0:
		return angle_var_avr/float(angle_plus_count)
	else:
		return None

#####

##ratio##

def speed_threshold_ratio(trajectory_datas):

	threshold = 0
	speed_list = trajectory_datas[:,2]
	count = 0
	for i in range(len(speed_list)):
		if speed_list[i] >= threshold:
			count = count+1
		else:
			count = count-1
	return count/float(len(speed_list))

def moving_avr_speed_threshold_ratio(trajectory_datas):

	threshold = 0.3
	speed_list = time_feature_moving_average_speed(trajectory_datas)
	count = 0
	for i in range(len(speed_list)):
		if speed_list[i] >= threshold:
			count = count+1
		else:
			count = count-1
	return count/float(len(speed_list))

def mvavr_speed_and_mvavr_angle_speed_threshold_ratio(trajectory_datas):

	speed_thresholds = [0.3,0.9]
	angle_threshold = 1.9
	speed_list = time_feature_moving_average_speed(trajectory_datas)
	angle_speed_list = time_feature_moving_average_angle_abs(trajectory_datas)
	count = 0
	for i in range(len(speed_list)):
		if speed_list[i] >= speed_thresholds[1]:
			count = count+1
		elif speed_list[i] <= speed_thresholds[0]:
			count = count-1
		else:
			if angle_speed_list[i]>=1.7 and angle_speed_list[i]<=2.3:
				count = count-1
			else:
				count = count+1

	return count/float(len(speed_list))

def moving_avr_angle_abs_threshold_ratio(trajectory_datas):

	threshold = 20
	angle_list = time_feature_moving_average_angle_abs(trajectory_datas)
	count = 0
	for i in range(len(angle_list)):
		if angle_list[i] >= threshold:
			count = count+1
		else:
			count = count-1
	return count/float(len(angle_list))

def moving_var_angle_abs_threshold_ratio(trajectory_datas):

	threshold = 2800
	angle_list = time_feature_moving_variance_angle_abs(trajectory_datas)
	count = 0
	for i in range(len(angle_list)):
		if angle_list[i] >= threshold:
			count = count+1
		else:
			count = count-1
	return count/float(len(angle_list))

def mv_avr_angle_abs_and_mv_avr_speed_threshold_ratio(trajectory_datas):

	threshold_speed = 0.3
	threshold_angle = 20
	speed_list = time_feature_moving_average_speed(trajectory_datas)
	angle_list = time_feature_moving_average_angle_abs(trajectory_datas)
	count = 0
	for i in range(len(speed_list)):
		if speed_list[i] >= threshold_speed:
			count = count+1#normal
		else:
			if angle_list[i] <= threshold_angle:
				count = count+1#normal
			else:
				count = count-1#dop

	return count/float(len(speed_list))

def stop_ratio(trajectory_datas):

	stop_speed_criteria = 0.2

	speed_list = trajectory_datas[:,2]
	count = 0
	for i in range(len(speed_list)):
		if speed_list[i] <= stop_speed_criteria:
			count = count+1

	return count/float(len(speed_list))

def sharp_speed_ratio(trajectory_datas):

	sharp_criteria = 0.3

	speed_list = trajectory_datas[:,2]
	max_speed = np.max(speed_list)
	min_speed = np.min(speed_list)
	speed_range = max_speed - min_speed
	max_count = 0
	min_count = 0
	for i in range(len(speed_list)):
		if speed_list[i] <= (min_speed+speed_range*sharp_criteria):
			min_count = min_count+1
		elif speed_list[i] >= (max_speed-speed_range*sharp_criteria):
			max_count = max_count+1

	return max_count/float(len(speed_list))

##others

def low_stop(trajectory_datas):

	speed_criteria = 0.0
	angle_speed_criteria = 1.0

	speed_list = trajectory_datas[:,2]
	angle_speed_list = trajectory_datas[:,1]
	count = 0
	for i in range(len(speed_list)):
		if (np.abs(angle_speed_list[i]) >= angle_speed_criteria):#(speed_list[i] <= speed_criteria) and
			count = count+1

	return count/float(len(speed_list))

def low_anglespeed_ratio(trajectory_datas):

	sharp_criteria = 0.5

	angle_speed_list = trajectory_datas[:,1]
	count = 0
	for i in range(len(angle_speed_list)):
		if np.abs(angle_speed_list[i]) <= sharp_criteria:
			count = count+1

	return count/float(len(angle_speed_list))

def sharp_acc_num(trajectory_datas):

	sharp_criteria = 0.4

	speed_list = trajectory_datas[:,2]
	acc_list = speed_list[3:] - speed_list[:-3]
	max_speed = np.max(speed_list)
	min_speed = np.min(speed_list)
	speed_range = max_speed - min_speed
	sharp_count = 0
	for i in range(len(acc_list)):
		if -acc_list[i] >= speed_range*sharp_criteria:
			sharp_count = sharp_count+1

	return sharp_count/float(len(acc_list))

def high_angle_speed_ratio(trajectory_datas):

	sharp_criteria = 0.4

	angle_speed_list = trajectory_datas[:,1]
	mean_angle_speed = np.mean(angle_speed_list)
	max_angle_speed = np.max(angle_speed_list)
	min_angle_speed = np.min(angle_speed_list)
	angle_speed_range = max_angle_speed - min_angle_speed
	sharp_count = 0
	for i in range(len(angle_speed_list)):
		if angle_speed_list[i] >= mean_angle_speed+angle_speed_range*sharp_criteria or angle_speed_list[i] <= mean_angle_speed-angle_speed_range*sharp_criteria:
			sharp_count = sharp_count+1

	return sharp_count/float(len(angle_speed_list))

def run_direction(trajectory_datas):

	x_point = trajectory_datas[3]
	y_point = trajectory_datas[4]

	directions = []
	for i in range(len(x_point)-1):
		current_x = x_point[i+1]-x_point[i]
		current_y = y_point[i+1]-y_point[i]
		current_direction = math.atan2(current_y, current_x)
		directions.append(current_direction)

	return np.mean(directions)
