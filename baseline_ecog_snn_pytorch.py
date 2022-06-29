import itertools
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#from evaluate_reservoir import *
#from utilis import *
from args import args as my_args
from evaluate_encoder import  *
from itertools import product
#import seaborn as sns
import time

if __name__ == '__main__':

  args = my_args()
  print(args.__dict__)
	# Fix the seed of all random number generator
  seed = 50
  random.seed(seed)
  np.random.seed(seed)
  df = pd.DataFrame({"recu":[],"epochs":[]})
  datasets=["bci3"]

  for i in range(len(datasets)):
    args.encode_thr_up = np.random.choice([1.36, 1, 0.5, 0.1])
    args.n_rec = np.random.choice([10,5,8,2,4])
    args.epochs=np.random.choice([20])
    args.lr=np.random.choice([1e-2,1e-1])
    args.loss = np.random.choice(['BCE', 'CE'])
    args.optimizer = np.random.choice(['SGD', 'Adam', 'NAG', 'RMSprop'])

    accuracy_epoch, loss_epoch = evaluate_encoder(args)
    df = df.append({"recu":args.n_rec,
            "epochs":args.epochs,
            "accuracy per epoch":accuracy_epoch,
            "loss per epoch":loss_epoch,
            "Learning Rate": args.lr,
            "Loss": args.loss,
            "Optimizer": args.optimizer,
            "Threshold": args.encode_thr_up
                    },ignore_index=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"eprop_"+str(int(args.seed))+".csv"
    pwd = os.getcwd()
    log_dir = pwd+'/log_dir/'
    df.to_csv(output_file, index=False)
