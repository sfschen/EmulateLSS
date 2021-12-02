#!/bin/bash
#SBATCH -p regular
#SBATCH -A cosmosim
#SBATCH -t 4:00:00
#SBATCH -J train_emu
#SBATCH -o train_emu
#SBATCH --mail-type=All
#SBATCH --mail-user  joe.derose13@gmail.com
#SBATCH -N 20
#SBATCH -C haswell
#SBATCH -L SCRATCH
#SBATCH --exclusive

#call salloc -N 20 -A cosmosim -q interactive -t 4:00:00 -C haswell -L SCRATCH to get interactive nodes for training data generation

module load python
source activate cobaya
export PYTHONPATH=${PYTHONPATH}:/global/project/projectdirs/desi/users/jderose/CobayaLSS/:/global/project/projectdirs/desi/users/jderose/CobayaLSS/lss_likelihood/ #replace with youre cobayalss directories
export HDF5_USE_FILE_LOCKING=FALSE

srun -n 640 python generate_training_data.py configs/ptchallenge_pkell_allpars_cmass2.yaml
