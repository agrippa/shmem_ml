#!/bin/bash -l

#SBATCH -J SHMEM_ML           # Job name
#SBATCH -p development        # Queue (partition) name
#SBATCH -N 8               # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=56
#SBATCH -t 00:10:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=max.grossman@gatech.edu
#SBATCH --exclusive
#####SBATCH --contiguous

set -e

ulimit -c unlimited

source ~/.profile

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $SHMEM_HOME"

# 2 sockets x 28 cores per socket for Frontera
export CORES_PER_SOCKET=28
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))
# export SHMEM_SYMMETRIC_SIZE=$((2 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
# export SHMEM_ML_POOL_SIZE=$((1 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
export SHMEM_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))
export SHMEM_ML_POOL_SIZE=$((512 * 1024 * 1024))
export SHMEM_ML_MAX_MAILBOX_BUFFERS=1
# export SHMEM_ML_HANG_ABORT=120
# export SHMEM_ML_HANG_ABORT_PE=0

mkdir -p $SCRATCH/job.$SLURM_JOB_ID
cd $SCRATCH/job.$SLURM_JOB_ID

#export OSHRUN_DEBUG=y
export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"

echo

oshrun -N 56 $HOME/shmem_ml/bin/shmem_ml_client_server $HOME/shmem_ml/example/python_client.py
# oshrun -N 2 $HOME/shmem_ml/bin/shmem_ml_client_server $HOME/shmem_ml/example/python_client.py
