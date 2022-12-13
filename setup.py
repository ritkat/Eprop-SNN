# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2020-2022 University of Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/

------------------------------------------------------------------------------
"""


from turtle import shape
import torch
import numpy as np
import numpy.random as rd
import sys
import os
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
from encode import *
from args import args as my_args

args = my_args()
nb_channels,spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list = encode(args)

class CueAccumulationDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type):
      
        n_cues     = 7
        f0         = 40
        t_cue      = 100
        t_wait     = 1200
        n_symbols  = 4
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 150
        self.seq_len    = n_cues*self.t_interval + t_wait
        self.n_in       = 40
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        
        if (type == 'train'):
            length = args.train_len
        else:
            length = args.test_len
            
    
        # Randomly assign group A and B
        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = rd.choice([0, 1], length)
        probs = np.zeros((length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]
    
        cue_assignments = np.zeros((length, n_cues), dtype=np.int)
        # For each example in batch, draw which cues are going to be active (left or right)
        for b in range(length):
            cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])
    
        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))
        t_silent = self.t_interval - t_cue
        for b in range(length):
            for k in range(n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[b, t_silent+k*self.t_interval:t_silent+k*self.t_interval+t_cue, c*n_channel:(c+1)*n_channel] = prob0
    
        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0
        input_spike_prob[:, :, 3*n_channel:] = prob0/4.
        input_spikes = generate_poisson_noise_np(input_spike_prob)
        self.x = torch.tensor(input_spikes).float()
    
        # Generate targets
        target_nums = np.zeros((length, self.seq_len), dtype=np.int)
        target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (self.seq_len, 1)))
        self.y = torch.tensor(target_nums).long()
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    if args.dataset == "cue_accumulation":
        print("=== Loading cue evidence accumulation dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cue_accumulation(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
        
    print("Training set length: "+str(args.full_train_len))
    print("Test set length: "+str(args.full_test_len))
    
    return (device, train_loader, traintest_loader, test_loader)


def load_dataset_cue_accumulation(args, kwargs):

    if args.datas=="bci3":

        nb_channels,spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list = encode(args)

        trainset = BCI3Dataset(spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, Y_Train_list,Y_Test_list)
        testset  = BCI3Dataset_test(spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, Y_Train_list,Y_Test_list)

        train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, **kwargs)
        traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       , **kwargs)
        test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       , **kwargs)
        
        args.n_classes      = trainset.n_out
        args.n_steps        = trainset.seq_len
        args.n_inputs       = trainset.n_in
        args.dt             = trainset.dt
        args.classif        = True
        args.full_train_len = len(trainset)
        args.full_test_len  = len(testset)
        args.delay_targets  = trainset.t_interval
        args.skip_test      = False

    else:
        trainset = CueAccumulationDataset(args,"train")
        testset  = CueAccumulationDataset(args,"test")

        train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, **kwargs)
        traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       , **kwargs)
        test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       , **kwargs)
        
        args.n_classes      = trainset.n_out
        args.n_steps        = trainset.seq_len
        args.n_inputs       = trainset.n_in
        args.dt             = trainset.dt
        args.classif        = True
        args.full_train_len = len(trainset)
        args.full_test_len  = len(testset)
        args.delay_targets  = trainset.t_interval
        args.skip_test      = False

    
    return (train_loader, traintest_loader, test_loader)


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes


##BCI3
class BCI3Dataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, Y_Train_list,Y_Test_list):
        #args=my_args()
        n_cues     = 7
        f0         = 128
        t_cue      = 100
        t_wait     = 1200
        n_symbols  = 4
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 150
        #self.seq_len    = n_cues*self.t_interval + t_wait
        self.seq_len    = 3000
        self.n_in       = 128
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        #data=np.load("dataset/bci3_encodedbci31111.npz", allow_pickle=True)
        spike_time_up_train=spike_times_train_up_list[0][0]
        spike_time_dn_train=spike_times_train_dn_list[0][0]
        spikes_train=spike_time_to_spike(spike_time_up_train, spike_time_dn_train, 3000)
        #spikes_train=np.array(spikes_train)
        shape=np.array(spikes_train.shape[1])
        events=Y_Train_list[0]
        for k in range(events.shape[0]):
            if events[k,0]==-1:
                events[k,0]=0

        for k in range(events.shape[0]):
            events[k]=events[k][0]

        repeated_event=np.array([])
        for k in range(events.shape[0]):
            if k==0:
                repeated_event=np.repeat(events[k],shape)
            else:
                repeated_event=np.vstack((repeated_event,np.repeat(events[k],shape) ))

        self.x=torch.tensor(spikes_train).float()
        self.y = torch.tensor(repeated_event).long()

        #sys.exit(0)
        '''
        1. np.load(file):  get spike times up and spike time down
        2. Convert the spike time array to spike array. This would be of length 3000, and would have value 1 
            index i,j if j.xyz ms is present in  spike time array of channel i. And zero otherwise. 
        3.Dimension of X (len,seq_len,nbinput) -> (nbSamples, nbTimepoints, nbChannels) -> (278,3000,64*2) 
            Shape the spike time up and dn, using (2) to generate X_train and X_test of size (nbttrainsamples,3000,64*2) and (nbtestsamples,3000,64*2)
        4.Dimension of Y (len,seq_len): (nbSamples, nbTimepoints)
            For our dataset,  y[sample_i,:] = label[sample_i]


        X is an array of 3000 samples.

        self.x = torch.tensor(X_Train).float()
        self.y = torch.tensor(y).long()
        '''

        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class BCI3Dataset_test(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, Y_Train_list,Y_Test_list):
        args=my_args()
        n_cues     = 7
        f0         = 128
        t_cue      = 100
        t_wait     = 1200
        n_symbols  = 4
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 150
        #self.seq_len    = n_cues*self.t_interval + t_wait
        self.seq_len    = 3000
        self.n_in       = 128
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        #data=np.load("dataset/bci3_encodedbci31111.npz", allow_pickle=True)
        #spikes_train=np.array(spikes_train)
        spike_time_up_test=spike_times_test_up_list[0][0]
        spike_time_dn_test=spike_times_test_dn_list[0][0]
        spikes_test=spike_time_to_spike(spike_time_up_test, spike_time_dn_test, 3000)
        #spikes_test=np.array(spikes_test)
        shape=np.array(spikes_test).shape[1]
        true_label=Y_Test_list[0]
        for k in range(true_label.shape[0]):
            if true_label[k,0]==-1:
                true_label[k,0]=0

        for k in range(true_label.shape[0]):
            true_label[k]=true_label[k][0]



        repeated_tlabel=np.array([])
        for k in range(0,true_label.shape[0]):
            if k==0:
                repeated_tlabel=np.repeat(true_label[k],shape)
            else:
                repeated_tlabel=np.vstack((repeated_tlabel,np.repeat(true_label[k],shape)))

        self.x=torch.tensor(spikes_test).float()
        self.y = torch.tensor(repeated_tlabel).long()
        #sys.exit(0)
        '''
        1. np.load(file):  get spike times up and spike time down
        2. Convert the spike time array to spike array. This would be of length 3000, and would have value 1 
            index i,j if j.xyz ms is present in  spike time array of channel i. And zero otherwise. 
        3.Dimension of X (len,seq_len,nbinput) -> (nbSamples, nbTimepoints, nbChannels) -> (278,3000,64*2) 
            Shape the spike time up and dn, using (2) to generate X_train and X_test of size (nbttrainsamples,3000,64*2) and (nbtestsamples,3000,64*2)
        4.Dimension of Y (len,seq_len): (nbSamples, nbTimepoints)
            For our dataset,  y[sample_i,:] = label[sample_i]


        X is an array of 3000 samples.

        self.x = torch.tensor(X_Train).float()
        self.y = torch.tensor(y).long()
        '''

        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
