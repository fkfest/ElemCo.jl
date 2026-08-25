#!/bin/bash

##########################################################
#
# SPECIFY PARAMETERS BELOW
#
##########################################################
#
# * Define path to VASP gamma-only binary:
#
#
#!/bin/bash -l
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

source /home/software/VASP/AMD/AOCL/LIB/5.1.0/gcc/amd-libs.cfg
source /usr/lib64/mpi/gcc/openmpi4/bin/mpivars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/software/VASP/AMD/HDF5_gcc14/hdf5_1.14.6_Install/lib
export PATH=/home/software/VASP/AMD/VASP_6.3.1_CC4S_gcc14/bin/:$PATH

# * Define path to VASP gamma-only binary:

VASP="mpirun -np 32 vasp_std"

# * Define energy cut-offs for the plane-wave basis sets:
#
enc=141
egw=100
#
# * Define basis set size. nbfp=11 corresponds to 10 virtual natural orbitals per occupied orbital.
#   Feel free to change the value for nbfp to check basis-set convergence of energies.
#
nbfp=11
#
# N.B.: A POTCAR file for boron and nitrogen needs to be present in the current directory
#
##########################################################

echo "++++++++++++++++++++++++++++++"
echo "Check if POTCAR file is present"
echo "++++++++++++++++++++++++++++++"


if test -f "POTCAR"; then
    echo "POTCAR file found."
else
    echo "POTCAR file not found. Please provide one for boron and nitrogen."
    exit
fi


echo "++++++++++++++++++++++++++++++"
echo "Preparing all input files for VASP calculations."
echo "++++++++++++++++++++++++++++++"

cat >KPOINTS<<!
Automatically generated mesh
       0
gamma
 1 1 1
 0 0 0
!

cat >POSCAR<<!
Li
3.5
       1.00         0.00         0.00
       0.00         1.00         0.00
       0.00         0.00         1.00
   Li
    2
Direct
     0.000000000         0.000000000         0.000000000
     0.500000000         0.500000000         0.500000000
!

echo "++++++++++++++++++++++++++++++"
echo "Going to run VASP calculation for the following POSCAR file"
echo "++++++++++++++++++++++++++++++"
cat POSCAR

echo "++++++++++++++++++++++++++++++"
echo "RUN HF"
echo "++++++++++++++++++++++++++++++"
cat >INCAR <<!
NELM=10000
ENCUT = $encut
ISMEAR = 0 ; SIGMA = 0.01
NCORE = 2
KPAR = 32
LHFCALC=.TRUE.
AEXX=1.0
HFRCUT=0
EDIFF = 1E-8
ALGO=C
PREC=A
ISYM = -1
LORBITALREAL = .TRUE.
LASPH=T
!
cat INCAR
$VASP
cp OUTCAR OUTCAR.HF


nb=`awk <OUTCAR "/maximum number of plane-waves:/ { print \\$5*2-1 }"`
echo "++++++++++++++++++++++++++++++"
echo "For the given plane-wave basis energy cut-off, there will be a total of " $nb " bands/orbitals."
echo "++++++++++++++++++++++++++++++"

echo "++++++++++++++++++++++++++++++"
echo "Compute Fock matrix and diagonalize in full basis"
echo "++++++++++++++++++++++++++++++"
cat >INCAR <<!
ENCUT = $enc
SIGMA=0.01
EDIFF = 1E-8
LHFCALC=.TRUE.
AEXX=1.0
ISYM=-1
ALGO = sub ; NELM = 1
NBANDS = $nb
!
cat INCAR
$VASP
cp OUTCAR OUTCAR.HFdiag

echo "++++++++++++++++++++++++++++++"
echo "Compute MP2 pair correlation energies in the CBS limit via extrapolation"
echo "++++++++++++++++++++++++++++++"
cat >INCAR <<!
ENCUT = $enc
SIGMA=0.01
LHFCALC=.TRUE.
AEXX=1.0
ISYM=-1
ALGO = MP2 
NBANDS = $nb
LSFACTOR=.TRUE.
!
rm WAVEDER
cat INCAR
$VASP
cp OUTCAR OUTCAR.MP2-CBS


echo "++++++++++++++++++++++++++++++"
echo "Compute approximate MP2 natural orbitals"
echo "see https://doi.org/10.1021/ct200263g for more details"
echo "The natural orbitals will be written to WAVECAR.FNO"
echo "++++++++++++++++++++++++++++++"
cat >INCAR <<!
ENCUT = $enc
SIGMA=0.01
LHFCALC=.TRUE.
AEXX=1.0
ISYM=-1
ALGO = MP2NO ;
NBANDS = $nb
LAPPROX=.TRUE.
!
rm WAVEDER
cat INCAR
$VASP
cp OUTCAR OUTCAR.MP2-NOs

nocc=`awk <OUTCAR "/NELEC/ { print \\$3/2 }"`

echo "++++++++++++++++++++++++++++++"
echo "This system contains $nocc occupied bands/orbitals."
echo "++++++++++++++++++++++++++++++"


echo "++++++++++++++++++++++++++++++"
echo "Preparing output for CC4S calculations."
echo "++++++++++++++++++++++++++++++"

nvirt=`echo "$nbfp - 1" | bc `
echo "Using " $nvirt " virtual orbitals per occupied orbital."

nbno=`awk <OUTCAR "/NELEC/ { print \\$3/2*$nbfp }"`
echo "Using " $nbno " bands/orbitals in total."

cp WAVECAR.FNO WAVECAR

echo "++++++++++++++++++++++++++++++"
echo "Recanonicalizing " $nbno "natural orbitals/bands using HF diagonalizer."
echo "++++++++++++++++++++++++++++++"
cat >INCAR <<!
ENCUT = $enc
SIGMA=0.01
EDIFF = 1E-8
LHFCALC=.TRUE.
AEXX=1.0
ISYM=-1
ALGO = sub ; NELM = 1
NBANDS = $nbno
NBANDSHIGH = $nbno
!
rm WAVEDER
cat INCAR
$VASP
cp OUTCAR OUTCAR.HFdiag-NOs

echo "++++++++++++++++++++++++++++++"
echo "Computing and writing CC4S output using " $nbno " bands."
echo "++++++++++++++++++++++++++++++"

cat >INCAR <<!
ENCUT = $enc
SIGMA=0.01
EDIFF = 1E-7
LHFCALC=.TRUE.
AEXX=1.0
ISYM=-1
ALGO=CC4S
NBANDS = $nbno
NBANDSHIGH = $nbno
ENCUTGW=$egw
ENCUTGWSOFT=$egw
!
cat INCAR
$VASP
cp OUTCAR OUTCAR.CC4S
