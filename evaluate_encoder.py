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

 "main.py" - Main file for training a spiking RNN with eligibility propagation (e-prop), as per the original paper:

        [G. Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons,"
         Nature communications, vol. 11, no. 3625, 2020]

    This code demonstrates e-prop on the cue accumulation task described in the e-prop paper by Bellec et al. The 
    e-prop implementation provided here only covers the LIF neuron model, which is presented by Bellec et al. as not
    able to learn the second-long time dependencies of the cue accumulation task.
    We solve this issue by using a second-long leakage time constant. We introduce this technique in the following paper:

        [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network
         processor enabling on-chip learning over second-long timescales," IEEE International Solid-State
         Circuits Conference (ISSCC), 2022]

    Example run: default parameters contained in this file provide suitable convergence for the cue accumulation task
             described in the e-prop paper. The code can just be ran with "python main.py"
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/
       
------------------------------------------------------------------------------
"""


#import argparse
import train
import setup
from args import args as my_args


def evaluate_encoder(args):
    #parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training')
    # General
    
    #args = parser.parse_args()

    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)    
    accuracy_epoch, loss_epoch = train.train(args, device, train_loader, traintest_loader, test_loader)

    return accuracy_epoch, loss_epoch
if __name__ == '__main__':
    args = my_args()
    accuracy_epoch, loss_epoch=evaluate_encoder(args)
    
    