#!/bin/bash
## Slurm job settings/options
#SBATCH --job-name=test_slurm
#SBATCH --output=job.out.%j
#SBATCH --error=job.err.%j
#SBATCH --mail-type=none
##SBATCH --partition=p.2h
#SBATCH --partition=gen3.48h
#SBATCH --nodes=1 
#SBATCH --exclusive
#SBATCH --time=01:00:00

echo "SLURM_NTASKS_PER_NODE:" $SLURM_NTASKS_PER_NODE
echo "SLURM_NTASKS:         " $SLURM_NTASKS

######################################
# Section for defining job variables and settings:
#
#
# /home/software/VASP/AMD/VASP_6.5.1/vasp_6.5.1_env.sh
module load gnu-openmpi/4.1.4
export OMP_NUM_THREADS=1

export CC4S_WORKDIR=/algpfs1/$USER/Slurmjob_$SLURM_JOB_ID

mkdir -p $CC4S_WORKDIR

# Preparing and moving inputfiles to tmp:

submitdir=$SLURM_SUBMIT_DIR

cp * $CC4S_WORKDIR
cd $CC4S_WORKDIR

mpirun -np 24 /home/software/VASP/AMD/Cc4s/cc4s/build/gcc-oblas-ompi/bin/Cc4s

cp * $submitdir

cd $submitdir
rm -r $CC4S_WORKDIR
echo $CC4S_WORKDIR" deleted."

echo "Job finished at"
date
################### Job Ended ###################
exit 0


