#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -J SHMEM_ML_PY
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $OPENSHMEM_HOME"

source ~/.bashrc.ext
setup_shmem

# module list

export LD_LIBRARY_PATH=$ARROW_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_HOME/lib64:$LD_LIBRARY_PATH

# 2 sockets x 16 cores per socket for Cori Haswell
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))
export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export SHMEM_ML_POOL_SIZE=$((512 * 1024 * 1024))
export PAT_RT_PERFCTR_MPX=1
export SHMEM_MAX_CTX=1

mkdir -p $SCRATCH/job.$SLURM_JOB_ID
cd $SCRATCH/job.$SLURM_JOB_ID

export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"

export PYTHONPATH=$HOME/shmem_ml:$PYTHONPATH
echo
time srun $SLURM_ARGS python $HOME/shmem_ml/example/python_wrapper.py
