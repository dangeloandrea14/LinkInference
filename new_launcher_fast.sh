#!/bin/bash
#SBATCH -J graph
#SBATCH -o output/logs/%j.out
#SBATCH -e output/logs/%j.out
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p cuda
#SBATCH --gres=gpu:fast
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python "$@"