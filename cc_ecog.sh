#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00 # DD-HH:MM:SS
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-1
#SBATCH --job-name=ecog_encoding

echo "Moving files"
cp -r $HOME/Eprop-SNN $SLURM_TMPDIR/Eprop-SNN
cd $SLURM_TMPDIR/Eprop-SNN

echo "Starting application"
mkdir -p "$HOME/eprop_results/"

if $HOME/env_snn/bin/python baseline_ecog_snn_pytorch.py --seed $SLURM_ARRAY_TASK_ID ; then
    echo "Copying results"
    mv "eprop_$SLURM_ARRAY_TASK_ID.csv" "$HOME/eprop_results/"
fi

wait