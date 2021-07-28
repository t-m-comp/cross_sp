import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

###txt###
def read_txt(filepath):
    f = open(filepath)
    data1 = f.read()
    f.close()
    lines1 = data1.split("\n")

    return lines1

#########

########for csv#################

def draw_graph(csv_data,yokojiku,tatejiku_list,name_or_num,output_name,option_Ylim=None):
    plt.close()
    if name_or_num=="num":

        yokojiku_line = csv_data[:,yokojiku].astype("float32")
        yokojiku_range = [np.min(yokojiku_line),np.max(yokojiku_line),np.max(yokojiku_line)-np.min(yokojiku_line)]#

        tatejiku_all = []
        for row_num in tatejiku_list:
            tatejiku_line = csv_data[:,row_num].astype("float32")
            tatejiku_all.append(tatejiku_line)
            plt.plot(yokojiku_line,tatejiku_line,label=str(row_num),linewidth=1)
        tatejiku_range = [np.min(tatejiku_all),np.max(tatejiku_all),np.max(tatejiku_all)-np.min(tatejiku_all)]

    elif name_or_num=="name":

        value_name_list = csv_data[0]
        yokojiku_line = csv_data[1:,np.where(value_name_list==yokojiku)[0][0]].astype("float32")
        yokojiku_range = [np.min(yokojiku_line),np.max(yokojiku_line),np.max(yokojiku_line)-np.min(yokojiku_line)]#

        tatejiku_all = []
        for row_name in tatejiku_list:
            tatejiku_line = csv_data[1:,np.where(value_name_list==row_name)[0][0]].astype("float32")
            tatejiku_all.append(tatejiku_line)
            plt.plot(yokojiku_line,tatejiku_line,label=row_name)
        tatejiku_range = [np.min(tatejiku_all),np.max(tatejiku_all),np.max(tatejiku_all)-np.min(tatejiku_all)]

    plt.legend()
    plt.xlim([yokojiku_range[0],yokojiku_range[1]])
    if option_Ylim!=None:
        plt.ylim([option_Ylim[0],option_Ylim[1]])
    else:
        plt.ylim([tatejiku_range[0]-tatejiku_range[2]*0.01,tatejiku_range[1]+tatejiku_range[2]*0.01])
    plt.savefig(output_name,dpi=100, bbox_inches='tight', pad_inches=0.1,linewidth=1)
    #plt.show()

def read_csv(filepath):
    #csv yomikomi
    flin = open(filepath, 'r')
    dataReader = csv.reader(flin)

    csv_data = []
    for line in dataReader:
        csv_data.append(line)
    flin.close()
    csv_data = np.array(csv_data)

    return csv_data

def read_csv_headcut(filepath):
    #csv yomikomi
    flin = open(filepath, 'r')
    dataReader = csv.reader(flin)

    csv_data = []
    for line in dataReader:
        csv_data.append(line)
    flin.close()
    csv_data = np.array(csv_data)

    return csv_data[1:]

def write_csv(filepath,csv_data):
    #csv kakikomi
    flout = open(filepath, 'w')
    writer = csv.writer(flout, lineterminator='\n')

    writer.writerows(csv_data)
    flout.close()

    return

def make_dirs(destination_dir):
    if os.path.exists(destination_dir)==False:
        os.makedirs(destination_dir)

    return
################################

########for sort################
def numpy_sort(numpy_matrix,column_num_for_sort,order_direction):

    sorted_matrix = numpy_matrix[numpy_matrix[:,column_num_for_sort].argsort()]

    if order_direction==1:
        return sorted_matrix
    elif order_direction==-1:
        #hanten
        return np.flipud(sorted_matrix)
    else:
        return sorted_matrix

def numpy_sort_addfunction(numpy_matrix,column_num_for_sort,order_direction,matrix_type):

    if matrix_type==1:
        header = numpy_matrix[0]
        matrix_main = numpy_matrix[1:].astype("float32")
        matrix_main = numpy_sort(matrix_main,column_num_for_sort,order_direction)
        return np.vstack((header,matrix_main))
    elif matrix_type==2:
        header = numpy_matrix[0]
        matrix_main = numpy_matrix[1:]
        matrix_main_left = matrix_main[:,0].reshape(-1,1)
        matrix_main_right = matrix_main[:,1:].astype("float32")
        matrix_main = np.hstack((matrix_main_left,matrix_main_right))

        matrix_main = numpy_sort(matrix_main,column_num_for_sort,order_direction)
        return np.vstack((header,matrix_main))
    elif matrix_type==3:
        header = numpy_matrix[0]
        matrix_main = numpy_matrix[1:]
        matrix_main_left = matrix_main[:,:2]
        matrix_main_right = matrix_main[:,2:].astype("float32")
        matrix_main = np.hstack((matrix_main_left,matrix_main_right))

        matrix_main = numpy_sort(matrix_main,column_num_for_sort,order_direction)
        return np.vstack((header,matrix_main))
    else:
        return numpy_sort(numpy_matrix,column_num_for_sort,order_direction)
