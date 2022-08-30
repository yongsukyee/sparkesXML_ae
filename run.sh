#!/bin/bash -l

#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=60GB

module load python
source ../env/bin/activate

# Train the model
srun -u python -u runalgo_ae.py

