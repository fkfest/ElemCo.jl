# ElemCo.jl <img align="right" src="files/coil.png" height=74> <br/><br/>

Julia implementation of various electron-correlation methods (main focus on coupled cluster methods).
The integrals are obtained from a FCIDUMP file or calculated using the `GaussianBasis` package.

## Capabilities

|          | Canonical          | DF               | Only closed-shell      |
|:----------|:--------------------:|:------------------:|:-----------------------:|
| RHF      |    :x:             |:heavy_check_mark:|:heavy_exclamation_mark:|
| UHF      |    :x:             |:heavy_check_mark:|                        |
| BO-HF    |:heavy_check_mark:  |                  |                        |
| MCSCF    |   :x:              |:wrench:          |                        |
| MP2      |:heavy_check_mark:  |                  |                        |
| CCSD     | :heavy_check_mark: |                  |:heavy_check_mark:      |
| RCCSD    | :heavy_check_mark: |                  |                        |
| UCCSD    | :heavy_check_mark: |                  |                        |
| ΛCCSD    | :heavy_check_mark: |                  |:heavy_check_mark:      |
| ΛUCCSD   | :heavy_check_mark: |                  |                        |
| CCSD(T)  | :heavy_check_mark: |                  |:heavy_check_mark:      |
| UCCSD(T) | :heavy_check_mark: |                  |                        |
| ΛCCSD(T) | :heavy_check_mark: |                  |:heavy_check_mark:      |
| ΛUCCSD(T)| :heavy_check_mark: |                  |                        |
| FR-CCSD  | :heavy_check_mark: |                  |                        |
| 2D-CCSD  | :heavy_check_mark: |                  |                        |
| DCSD     | :heavy_check_mark: |                  |:heavy_check_mark:      |
| RDCSD    | :heavy_check_mark: |                  |                        |
| UDCSD    | :heavy_check_mark: |                  |                        |
| ΛDCSD    | :heavy_check_mark: |                  |:heavy_check_mark:|
| ΛUDCSD   | :heavy_check_mark: |                  |                        |
| FR-DCSD  | :heavy_check_mark: |                  |                        |
| 2D-DCSD  | :heavy_check_mark: |                  |                        |
| SVD-DCSD | :heavy_check_mark: |:heavy_check_mark:|:heavy_exclamation_mark:|
| SVD-DC-CCSDT|:heavy_check_mark:|:heavy_check_mark:|:heavy_exclamation_mark:|


## Getting started

Requirements: julia (>1.8)

Packages: LinearAlgebra, NPZ, Mmap, TensorOperations, Printf, IterativeSolvers, GaussianBasis, DocStringExtensions, MKL(optional)

## Usage
For a development version of `ElemCo.jl`, clone the repository and create an alias to set the project to the `ElemCo.jl` directory,
```
alias jlm='julia --project=<path_to_ElemCo.jl>'
```
Now the command `jlm` can be used to start the calculations,
```
jlm input.jl
```

Default scratch directory path on Windows is the first environment variable found in the ordered list `TMP`, `TEMP`, `USERPROFILE`. 
On all other operating systems `TMPDIR`, `TMP`, `TEMP`, and `TEMPDIR`. If none of these are found, the path `/tmp` is used. 
Default scratch folder name is `elemcojlscr`. 

Variable names `fcidump`, `geometry` and `basis` are reserved for the file name of FCIDUMP, geometry specification and basis sets, respectively.

### Example
#### DCSD calculation using integrals from a FCIDUMP file
The ground state energy can be calculated using the DCSD method with the following script:
```julia
using ElemCo

fcidump = "../test/H2O.FCIDUMP"
@cc dcsd
```
#### DCSD calculation of the water molecule using density-fitted integrals
In order to calculate the ground state energy of the water molecule using the DCSD method, the following script can be used:
```julia
using ElemCo
geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")
@dfhf
@cc dcsd
```
The `@dfhf` macro calculates the density-fitted Hartree-Fock energy and orbitals 
and then DCSD calculation is performed using density-fitted integrals.

Further example scripts are provided in the `examples` directory.

Documentation is available at https://elem.co.il.

```
Electron coil
A poem by Bing

In the heart of an atom lies a tiny core
Where protons and neutrons are tightly bound
But around this nucleus, there's so much more
A cloud of electrons that swirls around

They don't orbit in circles like planets do
But jump and spin in quantum states
They can be here and there and everywhere too
And sometimes they even change their mates

This is the electron coil, the turmoil of the shell
The source of light and heat and power
The force that makes the atoms repel or gel
The spark that ignites the cosmic flower
```
