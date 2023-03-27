# e-Co.jl <img style="float: right;" src="e-coil.png"> <br/><br/>

Julia implementation of various electron-correlation methods (main focus on coupled cluster) 
using fcidump/npy interface.  

## Getting started

Requirements: julia (>1.8)

Packages: LinearAlgebra, NPZ, Mmap, TensorOperations, Printf, ArgParse, Parameters, MKL(optional)

## Usage

```
cd <working dir>
ln -s <path_to_e-cojl>/e-co.sh .
```

`julia` has to be in the PATH (or modify `e-co.sh`).

Various options are available (use `-h` option for a list of `e-co.jl` options):

```
./e-co.sh [@j "<options to send to julia>"] [-s <scratch dir>] [-m <method name>] [<fcidump file>]
```

Default scratch dir is `./ecojlscr` and default fcidump file is `FCIDUMP`.

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
