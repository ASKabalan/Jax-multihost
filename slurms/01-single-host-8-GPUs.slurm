#!/bin/bash
#SBATCH --account=XXX@v100
#SBATCH --job-name=01-single-Host-8-GPUs    # Name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
#SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
# Ici, reservation de 8x10=80 CPU (4 taches par noeud) et de 8 GPU (4 GPU par noeud) sur 2 noeuds :
#SBATCH --nodes=1                  # nombre de noeud
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4               # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:10:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=Single-Host-8-GPUs.out # name of output file 
#SBATCH --error=Single-Host-8-GPUs.out  # name of error file (here, in common with the output file)

module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd

# Loading modules
module load python
conda activate ax


module load nvidia-compilers/23.9
# Then do the following to configure the shell environment
export MODULEPATH=$NVHPC/modulefiles:$MODULEPATH
module load nvhpc
module load cuda/11.8.0  cmake cudnn/8.9.7.29-cuda 

# Echo of launched commands

set -x
# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun python $1