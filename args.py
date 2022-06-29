import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA training and run training on CPU')
    parser.add_argument('--dataset', type=str, choices = ['cue_accumulation'], default='cue_accumulation', help='Choice of the dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='Enables shuffling sample order in datasets after each epoch')
    parser.add_argument('--trials', type=int, default=1, help='Nomber of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices = ['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam', help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of the loss function (only for performance monitoring purposes, does not influence learning)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr-layer-norm', type=float, nargs='+', default=(0.05,0.05,1.0), help='Per-layer modulation factor of the learning rate')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training (limited by the available GPU memory)')
    parser.add_argument('--test-batch-size', type=int, default=5, help='Batch size for testing (limited by the available GPU memory)')
    parser.add_argument('--train-len', type=int, default=80, help='Number of training set samples')
    parser.add_argument('--test-len', type=int, default=20, help='Number of test set samples')
    parser.add_argument('--visualize', type=bool, default=True, help='Enable network visualization')
    parser.add_argument('--visualize-light', type=bool, default=True, help='Enable light mode in network visualization, plots traces only for a single neuron')
    # Network model parameters
    parser.add_argument('--n_rec', type=int, default=10, help='Number of recurrent units')
    parser.add_argument('--model', type=str, choices = ['LIF'], default='LIF', help='Neuron model in the recurrent layer. Support for the ALIF neuron model has been removed.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Firing threshold in the recurrent layer')
    parser.add_argument('--tau-mem', type=float, default=2000e-3, help='Membrane potential leakage time constant in the recurrent layer (in seconds)')
    parser.add_argument('--tau-out', type=float, default=20e-3, help='Membrane potential leakage time constant in the output layer (in seconds)')
    parser.add_argument('--bias-out', type=float, default=0.0, help='Bias of the output layer')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w-init-gain', type=float, nargs='+', default=(0.5,0.1,0.5), help='Gain parameter for the He Normal initialization of the input, recurrent and output layer weights')
    
    parser.add_argument(
        "--datas", default="bci3", type=str, help="Dataset(BCI3)"
    )

    parser.add_argument(
        "--encode_thr_up",
        default=1.36,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_thr_dn",
        default=1.36,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_refractory",
        default=1,
        type=float,
        help="Refractory period for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_interpfact",
        default=1,
        type=float,
        help="Interpolation factor in ms for spike encoding" "e.g. 1, 2, 3, etc.",
    )

    parser.add_argument(
        "--encoded_data_file_prefix",
        default='bci3_encoded',
        type=str,
        help="",
    )
    

    
    parser.add_argument(
        "--tstep",
        default=500,
        type=float,
        help="Readout layer step time in ms" "e.g. 200, 300, etc etc.",
    )


    parser.add_argument(
        "--tstart",
        default=0,
        type=float,
        help="Time point from which the simulated sub-segment(of length tstep) is used as a feature for readout layer" ">0 (in ms).",
    )

    parser.add_argument(
        "--tlast",
        default=3000,
        type=float,
        help="Time point till which the simulated sub-segment(of length tstep) is used as a feature for readout layer" "e.g. <1800> (in ms).",
    )
    parser.add_argument(
        "--duration",
        default=3000,
        type=float,
        help="Time point till which the simulation has to be run",
    )

    parser.add_argument(
        "--preprocess",
        default=1,
        type=int,
        help="1 = Preprocessing has to be done, 0 = No Preprocessing",
    )

    parser.add_argument(
        "--seed",
        default=50,
        type=float,
        help="Seed for random number generation",
    )


    parser.add_argument('--experiment_name', default='standalone', type=str,
                        help='Name for identifying the experiment'
                               'e.g. plot ')


    parser.add_argument('--fold', default=3, type=float,
                        help='Fold for train/test'
                             'e.g. 1, 2, 3 ')
    
    parser.add_argument('--population', default=300, type=int,
                        help='population size for genetic search'
                             'e.g. 100, 200, 300 ')
  
    parser.add_argument('--f_split', default=2, type=int,
                    help='Splitting of time points'
                          'Use case: if 2, then 0-1500 and 1500-3000')
    
    parser.add_argument('--modes', default=[], type=list,
                    help='modes to calculate'
                          'e.g. ["genetic"]')

    parser.add_argument('--niter', default=100, type=int,
                    help='Numner of iterations for random search'
                          'e.g. 50, 100, 200 ')

    parser.add_argument('--scaler', default="Standard", type=str,
                    help='Type to scaler to nomralise the data'
                          'e.g. Standard, MinMax ')
    
    parser.add_argument("--kfold",default=3,type=int,
        help="number of folds in kfold",
    )
    parser.add_argument("--maxft",default=5,type=int,
        help="upper limit on number of features taken in genetic search",
    )
    parser.add_argument(
        "--encode_thr_dev",
        default=1.36,
        type=float,
        help="Standard deviation for threshold distribution",
    )

    parser.add_argument(
        "--encode_window",
        default=1000,
        type=float,
        help="Time window for MW for spike encoding" "e.g. 1000, 2000.",
    )

    parser.add_argument(
        "--method", default="tc", type=str, help="tc/mw/sf"
    )

    parser.add_argument('--log_file_path', default=None, 
                        help='Path for log file')

    my_args = parser.parse_args()

    return my_args