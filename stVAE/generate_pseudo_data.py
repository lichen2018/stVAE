from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import random
from random import shuffle
import gc
import argparse
import sys
import multiprocessing

import csv
import os





def generate_mixture_squences(unique_label_list, sample_seq_dict, batch_idx):
    composition_cell_list = []
    composition_cell_label_list = []
    for idx in range(5000):
        cell_types_num = 2
        cell_type_list = random.sample(unique_label_list, cell_types_num)
        compostion_label = [0]*len(unique_label_list)

        for count in range(50):
            tmp_sum = 0
            cell_composition_list = []
            for cell_type in cell_type_list:
                select_num = random.randint(1,2)
                cell_composition_list.extend(random.sample(list(sample_seq_dict[cell_type]), select_num))
                ct_idx = unique_label_list.index(cell_type)
                compostion_label[ct_idx] = select_num
                tmp_sum += select_num
            cell_composition_array = np.array([np.array(cell) for cell in cell_composition_list])
            composition_cell = np.sum(cell_composition_array, axis=0)//tmp_sum
            compostion_label_array = np.array(compostion_label)/tmp_sum

            composition_cell_list.append(composition_cell)
            composition_cell_label_list.append(compostion_label_array)
    data_file = './data_'+str(batch_idx)+'.csv'
    label_file = './label_'+str(batch_idx)+'.csv'
    pd.DataFrame(np.array(composition_cell_list)).to_csv(data_file)
    pd.DataFrame(np.array(composition_cell_label_list)).to_csv(label_file)
    composition_cell_list = []
    composition_cell_label_list = []
    gc.collect()


def loadTrainSeq(batch_idx,scRNA_file, scRNA_label_file):
    expr_df = pd.read_csv(scRNA_file, delimiter = ',', header = 0, index_col = 0)
    label_df = pd.read_csv(scRNA_label_file, delimiter = ',', header = 0, index_col = 0)
    sample_seq_dict = {}
    unique_label_list = []
    ct_name_lst = list(label_df.columns)
    for ct in ct_name_lst:
        unique_label_list.append(ct)
        spot_index_lst = np.nonzero(np.array(label_df.loc[:,ct]))[0]
        spot_expr = expr_df.values[spot_index_lst,:]
        sample_seq_dict.update({ct : spot_expr})
        #ct_num_selected_dict.update({ct : 0})

    generate_mixture_squences(unique_label_list, sample_seq_dict, batch_idx)



def worker(batch_idx, scRNA_file, scRNA_label_file):
    loadTrainSeq(batch_idx, scRNA_file, scRNA_label_file)


def generate_pseudo_spots(scRNA_file, scRNA_label_file):
    procs = []
    batch_idx = 0
    while batch_idx < 2:
        p = multiprocessing.Process(target=worker, args= (batch_idx, scRNA_file, scRNA_label_file,))
        print(str(batch_idx)+'processing!')
        procs.append(p)
        p.start()
        print(p.pid)
        batch_idx += 1
    for proc in procs:
        proc.join()


def generate_train_valid_batches(scRNA_file='scRNA.csv', scRNA_label_file='scRNA_label.csv', pseudo_data_path= './batch_data/'):
    generate_pseudo_spots(scRNA_file, scRNA_label_file)
    id_label_list = []
    for i in range(2):
        file_path = './label_'+str(i)+'.csv'
        label_df = pd.read_csv(file_path,delimiter=',',header=0,index_col=0)
        count = 0
        for (cell,rowData) in label_df.iterrows():
            sample_id = str(i) + '_' + str(count)
            id_label_list.append([sample_id, np.asarray(rowData)])
            count += 1

    shuffle(id_label_list)
    train_label_list = id_label_list[:int(0.8*len(id_label_list))]
    validate_label_list = id_label_list[int(0.8*len(id_label_list)):]

    sample_location_dict = {}
    batch_size = 120
    for i in range(len(train_label_list)):
        batach_name = 'train_data_batch_'+str(int(i/batch_size))+'.csv'
        sample_location_dict.update({train_label_list[i][0]:[batach_name,train_label_list[i][1]]})

    for i in range(len(validate_label_list)):
        batach_name = 'validate_data_batch_'+str(int(i/batch_size))+'.csv'
        sample_location_dict.update({validate_label_list[i][0]:[batach_name,validate_label_list[i][1]]})


    if os.path.exists(pseudo_data_path):
        os.system("rm -rf "+pseudo_data_path+"*csv")
    else:
        os.makedirs(pseudo_data_path)
    data_list = []
    label_list = []
    for i in range(2):
        data_list = []
        label_list = []
        data_file_path = './data_'+str(i)+'.csv'
        data_df = pd.read_csv(data_file_path,delimiter=',',header=0,index_col=0)
        for (cell,rowData) in data_df.iterrows():
            data_list.append(np.asarray(rowData))
        label_file_path = './label_'+str(i)+'.csv'
        label_df = pd.read_csv(label_file_path,delimiter=',',header=0,index_col=0)
        for (cell,rowData) in label_df.iterrows():
            label_list.append(np.asarray(rowData))
        for j in range(len(data_list)):
            sample_id = str(i) + '_' + str(j)
            data_batch_file = sample_location_dict[sample_id][0]
            with open(pseudo_data_path+data_batch_file, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(data_list[j])
                f.close()
            label_batch_file = data_batch_file.replace('data','label')
            with open(pseudo_data_path+label_batch_file, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(label_list[j])
                f.close()
            if list(sample_location_dict[sample_id][1]) != list(label_list[j]):
                flag = True
        gc.collect()






