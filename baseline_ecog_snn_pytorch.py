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

  parameters = dict(
		rec_units = [10],
    epochs=[2,3,5],
    lr=[1e-4,1e-3,1e-2,1e-1]
    )
  param_values = [v for v in parameters.values()] 
  for args.n_rec, args.epochs, args.lr in product(*param_values):
    accuracy_epoch, loss_epoch = evaluate_encoder(args)
    df = df.append({"recu":args.n_rec,
            "epochs":args.epochs,
            "accuracy per epoch":accuracy_epoch,
            "loss per epoch":loss_epoch,
            "Learning Rate": args.lr
                    },ignore_index=True)



    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"eprop_"+str(int(args.seed))+".csv"
    pwd = os.getcwd()
    log_dir = pwd+'/log_dir/'
    df.to_csv(output_file, index=False)
