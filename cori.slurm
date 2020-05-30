#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:05:00
#SBATCH -J SHMEM_ML
#SBATCH --exclusive
###SBATCH --contiguous
#SBATCH --mail-type=ALL

set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $OPENSHMEM_INSTALL"

export LD_LIBRARY_PATH=$ARROW_HOME/lib64:$LD_LIBRARY_PATH

# 2 sockets x 16 cores per socket for Cori Haswell
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))
export SHMEM_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))

cd $HOME/shmem_ml
make clean
make -j bin/save_load_get_set bin/bfs

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"

echo "save_load_get_set"
srun $SLURM_ARGS $HOME/shmem_ml/bin/save_load_get_set
echo
echo "bfs"
srun $SLURM_ARGS $HOME/shmem_ml/bin/bfs $SCRATCH/job.$SLURM_JOB_ID/edges.bin
echo

