#!/usr/bin/env python
# coding: utf-8

# spike conversion algorithm.
# Spike time array contains values of spike times in ms.
# Saved arrays :

# X: Array of the EMG/EEG/ECoG Digital time series data with length = 200
# Y: Array of the labels of theing data with length = 200

# spike_times_up: Spike time arrays with upward polarity in ms for X. length = 200
# spike_times_dn: Spike time arrays with downward polarity in ms for X. length = 200

# Author : Nikhil Garg, 3IT Sherbrooke ; nikhilgarg.bits@gmail.com
# Created : 15 July 2020
# Last edited : 3rd January 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import butter, lfilter, welch, square  # for signal filtering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from utilis import *
from args import args as my_args

def encode(args):
    # general stuff
    # sampling frequency of MYO
    
    VERBOSE = True
    pwd = os. getcwd()
    # Selecting the directory of the datasets and the number of channels
    if args.datas == "bci3":
        data_dir = pwd + "/dataset/bci3.npz"
        fs = 1000
        nb_channels = 64
        
    elif args.datas =="zt_mot":
      data_dir = pwd + "/dataset/zt_mot_epochs.npz"
      nb_channels=48
      fs = 1000

    elif args.datas =="jc_mot":
      data_dir = pwd + "/dataset/jc_mot_epochs.npz"
      nb_channels=48
      fs = 1000

    elif args.datas =="fp_mot":
      data_dir = pwd + "/dataset/fp_mot_epochs.npz"
      nb_channels=62
      fs = 1000

    elif args.datas =="ca_mot":
        data_dir = pwd + "/dataset/ca_mot_epochs.npz"
        nb_channels=59
        fs = 1000
        
    elif args.datas =="jp_mot":
        data_dir = pwd + "/dataset/jp_mot_epochs.npz"
        nb_channels=58
        fs = 1000

    elif args.datas =="jm_mot":
        data_dir = pwd + "/dataset/jm_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.datas =="hh_mot":
        data_dir = pwd + "/dataset/hh_mot_epochs.npz"
        nb_channels=41
        fs = 1000   

    elif args.datas =="hl_mot":
        data_dir = pwd + "/dataset/hl_mot_epochs.npz"
        nb_channels=64
        fs = 1000  

    elif args.datas =="gc_mot":
        data_dir = pwd + "/dataset/gc_mot_epochs.npz"
        nb_channels=64
        fs = 1000  

    elif args.datas =="ug_mot":
        data_dir = pwd + "/dataset/ug_mot_epochs.npz"
        nb_channels=25
        fs = 1000  

    elif args.datas =="wc_mot":
        data_dir = pwd + "/dataset/wc_mot_epochs.npz"
        nb_channels=64
        fs = 1000


    elif args.datas =="jf_mot":
        data_dir = pwd + "/dataset/jf_mot_epochs.npz"
        nb_channels=39
        fs = 1000

    elif args.datas =="bp_mot":
        data_dir = pwd + "/dataset/bp_mot_epochs.npz"
        nb_channels=47
        fs = 1000

    elif args.datas =="de_mot":
        data_dir = pwd + "/dataset/de_mot_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.datas =="cc_mot":
        data_dir = pwd + "/dataset/cc_mot_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.datas =="rr_mot":
        data_dir = pwd + "/dataset/rr_mot_epochs.npz"
        nb_channels=49
        fs = 1000

    elif args.datas =="jt_mot":
        data_dir = pwd + "/dataset/jt_mot_epochs.npz"
        nb_channels=62
        fs = 1000

    elif args.datas =="gf_mot":
        data_dir = pwd + "/dataset/gf_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.datas =="rh_mot":
        data_dir = pwd + "/dataset/rh_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.datas =="rr_im":
        data_dir = pwd + "/dataset/rr_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.datas =="jc_im":
        data_dir = pwd + "/dataset/jc_im_epochs.npz"
        nb_channels=48
        fs = 1000

    elif args.datas =="jm_im":
        data_dir = pwd + "/dataset/jm_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.datas =="fp_im":
        data_dir = pwd + "/dataset/fp_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.datas =="bp_im":
        data_dir = pwd + "/dataset/bp_im_epochs.npz"
        nb_channels=46
        fs = 1000

    elif args.datas =="rh_im":
        data_dir = pwd + "/dataset/rh_im_epochs.npz"
        nb_channels=64
        fs = 1000

    spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list=[], [],[], [],[], [],[], [],[]

    if args.datas=="bci3":
        #Add data here
        X_Train = []
        Y_Train = []
        X_Test = []
        Y_Test = []

        data = np.load(data_dir)
        X_Train = data['X_Train']
        # Shape of X_Train is (278, 64, 3000)
        Y_Train = data['Y_Train']

        X_Test = data['X_Test']
        # Shape of X_Test is (100, 64, 3000)
        Y_Test = data['Y_Test']

        Y_Train_list.append(Y_Train)
        Y_Test_list.append(Y_Test)



        X_Train = np.array(X_Train)
        Y_Train = np.array(Y_Train)

        X_Test = np.array(X_Test)
        Y_Test = np.array(Y_Test)

        
        training_data=X_Train
        testing_data=X_Test

        if(args.preprocess==1):
            # Common Average Referencing 
            for j in range(0, X_Train.shape[0]):
                car=np.zeros((training_data.shape[2],))
                for i in range(0, X_Train.shape[1]):
                    car= car + training_data[j,i,:]
                car=car/X_Train.shape[1]
                #car.shape
                for k in range(0, X_Train.shape[1]):
                    training_data[j,k,:]=training_data[j,k,:]-car
    
            for j in range(0, testing_data.shape[0]):
                car=np.zeros((testing_data.shape[2],))
                for i in range(0, testing_data.shape[1]):
                    car= car + testing_data[j,i,:]
                car=car/testing_data.shape[1]
                #car.shape
                for k in range(0, testing_data.shape[1]):
                    testing_data[j,k,:]=testing_data[j,k,:]-car

            #Standard Scaler

            for j in range(0, training_data.shape[0]):
                kr=training_data[j,:,:]
                kr=training_data[j,:,:]
                if args.scaler=="Standard":
                    scaler=StandardScaler()
                if args.scaler=="minmax":
                    scaler=MinMaxScaler()
                scaled=scaler.fit(kr.T)
                training_data[j,:,:]=scaled.transform(kr.T).T
                
            for j in range(0, testing_data.shape[0]):
                kr=testing_data[j,:,:]
                kr=testing_data[j,:,:]
                if args.scaler=="Standard":
                    scaler=StandardScaler()
                if args.scaler=="minmax":
                    scaler=MinMaxScaler()
                scaled=scaler.fit(kr.T)
                testing_data[j,:,:]=scaled.transform(kr.T).T
                
        training_data = np.moveaxis(training_data, 2, 1)
        X_Train_list.append(training_data)

        #Shape now is (278, 3000, 64)

        testing_data = np.moveaxis(testing_data, 2, 1)
        X_Test_list.append(testing_data)

        interpfact = args.encode_interpfact
        refractory_period = args.encode_refractory  # in ms
        th_up = args.encode_thr_up
        th_dn = args.encode_thr_dn
        f_split=args.f_split
        # Number of parts that the 3000 segment would be split in. For eg: if f_split=2 then parts is 0,1500 ; 1500,3000
        parts=training_data.shape[1]/f_split
        # Make a list that stores the partitioned array. For eg: X_Train_s[0].shape:(278,1500,64)
        X_Train_s=[]
        X_Test_s=[]
        for h in range(f_split):
            X_Train_s.append(training_data[:,h*int(parts):(h+1)*int(parts),:])
            X_Test_s.append(testing_data[:,h*int(parts):(h+1)*int(parts),:])

        spike_times_train_up_l=[]
        spike_times_train_dn_l=[]
        # Generate the data
        for h in range(f_split):
            X=X_Train_s[h]
            Y=Y_Train
            spike_times_train_up = []
            spike_times_train_dn = []
            for i in range(len(X)):
                spk_up, spk_dn = gen_spike_time(
                    time_series_data=X[i],
                    interpfact=interpfact,
                    fs=fs,
                    th_up=th_up,
                    th_dn=th_dn,
                    refractory_period=refractory_period,
                    dev=args.encode_thr_dev,
                    method=args.method,
                    window=args.encode_window
                )
                spike_times_train_up.append(spk_up)
                spike_times_train_dn.append(spk_dn)
            spike_times_train_up_l.append(spike_times_train_up)
            spike_times_train_dn_l.append(spike_times_train_dn)
        spike_times_train_up_list.append(spike_times_train_up_l)
        spike_times_train_dn_list.append(spike_times_train_dn_l)

        
        # Need to be looked upon in further iterations
        rate_up = gen_spike_rate(spike_times_train_up)
        rate_dn = gen_spike_rate(spike_times_train_dn)
        avg_spike_rate = (rate_up+rate_dn)/2
        avg_spike_rate_list.append(avg_spike_rate)
        print("Average spiking rate")
        print(avg_spike_rate)

        # Generate the  data
        spike_times_test_up_l=[]
        spike_times_test_dn_l=[]
        for h in range(f_split):
            X=X_Test_s[h]
            Y=Y_Test
            spike_times_test_up = []
            spike_times_test_dn = []
            for i in range(len(X)):
                spk_up, spk_dn = gen_spike_time(
                    time_series_data=X[i],
                    interpfact=interpfact,
                    fs=fs,
                    th_up=th_up,
                    th_dn=th_dn,
                    refractory_period=refractory_period,
                    dev=args.encode_thr_dev,
                    method=args.method,
                    window=args.encode_window
                )
                spike_times_test_up.append(spk_up)
                spike_times_test_dn.append(spk_dn)
            spike_times_test_up_l.append(spike_times_test_up)
            spike_times_test_dn_l.append(spike_times_test_dn)
        spike_times_test_up_list.append(spike_times_test_up_l)
        spike_times_test_dn_list.append(spike_times_test_dn_l)
        
            
        # print(len(X))
        print("Number of test samples in dataset:")
        print(len(X_Test))
        print(len(Y_Test))
        # print("Class labels:")
        # print(list(set(Y_Test)))


        spike_times_train_up = np.array(spike_times_train_up)
        spike_times_test_up = np.array(spike_times_test_up)
        spike_times_train_dn = np.array(spike_times_train_dn)
        spike_times_test_dn = np.array(spike_times_test_dn)


        file_path = "dataset/"
        file_name = args.encoded_data_file_prefix + str(args.datas) + str(args.encode_thr_up) + str(
            args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) + ".npz"

        np.savez_compressed(
            file_path + file_name,
            Y_Train=Y_Train,
            Y_Test=Y_Test,
            spike_times_train_up=spike_times_train_up,
            spike_times_train_dn=spike_times_train_dn,
            spike_times_test_up=spike_times_test_up,
            spike_times_test_dn=spike_times_test_dn,
        )
        return nb_channels,spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list

    else:
        X_Train = []
        Y_Train = []
        X_Test = []
        Y_Test = []

        data = np.load(data_dir)
        X_Train = data['X']
        Y_Train = data['y']

        # Splitting the data into k folds 
        kf3 = KFold(n_splits=args.kfold, shuffle=False)
        for train_index, test_index in kf3.split(X_Train):
            X_Tr = X_Train[train_index]
            Y_Tr = Y_Train[train_index]
            X_Test = X_Train[test_index]
            Y_Test = Y_Train[test_index]

            # Storing the for K-Fold Cross Validation datasets
            Y_Train_list.append(Y_Tr)
            Y_Test_list.append(Y_Test)


            training_data=X_Tr
            testing_data=X_Test
            if(args.preprocess==1):
                

                #training_data.shape
                for j in range(0, X_Tr.shape[0]):
                    car=np.zeros((training_data.shape[2],))
                    for i in range(0, X_Tr.shape[1]):
                        car= car + training_data[j,i,:]
                    car=car/X_Tr.shape[1]
                    #car.shape
                    for k in range(0, X_Tr.shape[1]):
                        training_data[j,k,:]=training_data[j,k,:]-car
                
                
                for j in range(0, testing_data.shape[0]):
                    car=np.zeros((testing_data.shape[2],))
                    for i in range(0, testing_data.shape[1]):
                        car= car + testing_data[j,i,:]
                    car=car/testing_data.shape[1]
                    #car.shape
                    for k in range(0, testing_data.shape[1]):
                        testing_data[j,k,:]=testing_data[j,k,:]-car

                #Standard Scaler

                for j in range(0, training_data.shape[0]):
                    kr=training_data[j,:,:]
                    kr=training_data[j,:,:]
                    if args.scaler=="Standard":
                        scaler=StandardScaler()
                    if args.scaler=="minmax":
                        scaler=MinMaxScaler()
                    scaled=scaler.fit(kr.T)
                    training_data[j,:,:]=scaled.transform(kr.T).T
                    

                for j in range(0, testing_data.shape[0]):
                    kr=testing_data[j,:,:]
                    kr=testing_data[j,:,:]
                    if args.scaler=="Standard":
                        scaler=StandardScaler()
                    if args.scaler=="minmax":
                        scaler=MinMaxScaler()
                    scaled=scaler.fit(kr.T)
                    testing_data[j,:,:]=scaled.transform(kr.T).T


            training_data= np.moveaxis(training_data, 2, 1)
            testing_data = np.moveaxis(testing_data, 2, 1)
            X_Train_list.append(training_data)
            X_Test_list.append(testing_data)

            interpfact = args.encode_interpfact
            refractory_period = args.encode_refractory  # in ms
            th_up = args.encode_thr_up
            th_dn = args.encode_thr_dn
            f_split=args.f_split
            #no. of parts that the 3000 segment would be split in. For eg: if f_split=2 then parts is 0,1500 ; 1500,3000
            parts=training_data.shape[1]/f_split
            #make a list that stores the partitioned array. For eg: X_Train_s[0].shape:(278,1500,64)
            X_Train_s=[]
            X_Test_s=[]
            for h in range(f_split):
                X_Train_s.append(training_data[:,h*int(parts):(h+1)*int(parts),:])
                X_Test_s.append(testing_data[:,h*int(parts):(h+1)*int(parts),:])


            # Generate the  data
            spike_times_train_up_l=[]
            spike_times_train_dn_l=[]
            # Generate the  data
            for h in range(f_split):
                X=X_Train_s[h]
                Y=Y_Train
                spike_times_train_up = []
                spike_times_train_dn = []
                for i in range(len(X)):
                    spk_up, spk_dn = gen_spike_time(
                        time_series_data=X[i],
                        interpfact=interpfact,
                        fs=fs,
                        th_up=th_up,
                        th_dn=th_dn,
                        refractory_period=refractory_period,
                        dev=args.encode_thr_dev,
                        method=args.method,
                        window=args.encode_window
                    )
                    spike_times_train_up.append(spk_up)
                    spike_times_train_dn.append(spk_dn)
                spike_times_train_up_l.append(spike_times_train_up)
                spike_times_train_dn_l.append(spike_times_train_dn)
            spike_times_train_up_list.append(spike_times_train_up_l)
            spike_times_train_dn_list.append(spike_times_train_dn_l)

            rate_up = gen_spike_rate(spike_times_train_up)
            rate_dn = gen_spike_rate(spike_times_train_dn)
            avg_spike_rate = (rate_up+rate_dn)/2
            avg_spike_rate_list.append(avg_spike_rate)

                # Generate the  data
            spike_times_test_up_l=[]
            spike_times_test_dn_l=[]
            for h in range(f_split):
                X=X_Test_s[h]
                Y=Y_Test
                spike_times_test_up = []
                spike_times_test_dn = []
                for i in range(len(X)):
                    spk_up, spk_dn = gen_spike_time(
                        time_series_data=X[i],
                        interpfact=interpfact,
                        fs=fs,
                        th_up=th_up,
                        th_dn=th_dn,
                        refractory_period=refractory_period,
                        dev=args.encode_thr_dev,
                        method=args.method,
                        window=args.encode_window
                    )
                    spike_times_test_up.append(spk_up)
                    spike_times_test_dn.append(spk_dn)
                spike_times_test_up_l.append(spike_times_test_up)
                spike_times_test_dn_l.append(spike_times_test_dn)
            spike_times_test_up_list.append(spike_times_test_up_l)
            spike_times_test_dn_list.append(spike_times_test_dn_l)

            spike_times_train_up = np.array(spike_times_train_up)
            spike_times_test_up = np.array(spike_times_test_up)
            spike_times_train_dn = np.array(spike_times_train_dn)
            spike_times_test_dn = np.array(spike_times_test_dn)

            file_path = "dataset/"
            file_name = args.encoded_data_file_prefix + str(args.datas) + str(args.encode_thr_up) + str(
                args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) + ".npz"

            np.savez_compressed(
                file_path + file_name,
                Y_Train=Y_Tr,
                Y_Test=Y_Test,
                spike_times_train_up=spike_times_train_up,
                spike_times_train_dn=spike_times_train_dn,
                spike_times_test_up=spike_times_test_up,
                spike_times_test_dn=spike_times_test_dn,
            )
        return nb_channels,spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list


if __name__ == '__main__':
    args = my_args()
    print(args.__dict__)
    # Fix the seed of all random number generator
    encode(args)
