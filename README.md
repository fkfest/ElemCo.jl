# e-Co.jl

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

