@doc raw"""
This module provides an interface to Molpro to read and write orbitals and other data.

It includes functions to read Molpro matrop files to import overlap/density matrices and 
orbital coefficients.

A convenient way to use the interface is given below.

## ElemCo.jl Molpro Interface: Seamless Julia-Molpro Integration

The ElemCo.jl package provides a streamlined workflow for integrating Julia calculations with 
Molpro quantum chemistry computations through a simple include mechanism and macro-based data exchange.

### Setup

Add the following line to your Molpro input file:
```molpro
include,elemcoil
```

This includes a separate configuration file `elemcoil` that handles all data exchange between
Molpro and Julia.

### Configuration File (`elemcoil`)

The `elemcoil` file defines the interface between Molpro and Julia:

```molpro
$XML='data/mol.xml'
$ORBITALS='data/orbs.dat'
!$ECORBITALS='data/ecorbs.dat'
$ECVARIABLES='data/ecvars.dat'

system,'mkdir -p','data'

put,xml,$XML

{matrop
load ORB ORB
write,ORB,$ORBITALS,new,sci
}

system,'julia input.jl',' > elemcoil.log'

!{matrop
!read,ORB,file=$ECORBITALS
!save,ORB,3200.2,ORBITALS
!}

readvar,$ECVARIABLES
```

**Key Components:**
- **Variable definitions**: Define file paths for XML output, orbital data, and variable exchange
- **Data export**: Automatically exports Molpro XML and orbital matrices to specified files
- **Julia execution**: Calls the Julia script `input.jl` and logs output to `elemcoil.log`
- **Data import**: Reads back modified variables and orbitals (commented sections for optional use)

### Julia Script (`input.jl`)

Use the convenient `@molpro_input` and `@molpro_output` macros for seamless data access:

```julia
using ElemCo

# Read Molpro data using the @molpro_input macro
@molpro_input

# Perform Julia calculations
result = @cc dcsd

# Export results back to Molpro using @molpro_output macro, with a prefix for variable names
@molpro_output result prefix="EC_" 
```

This script reads the Molpro data defined in `elemcoil`, performs calculations using Julia's
ElemCo.jl package, and exports results back to Molpro with a specified prefix.

The geometry and basis set information is automatically handled by the `@molpro_input` macro.
Note that the XML file in Molpro stores only the AO basis set. 
The fitting basis set is not stored in the XML file. If you need a specific fitting basis set,
you can define it *before* the `@molpro_input` macro in the usual way, e.g.:
```julia
geometry = Dict("mpfit" => "avtz-mpfit", "jkfit" => "vqz-jkfit")
@molpro_input
```

### Workflow Benefits

1. **Automatic data handling**: No manual file path management - everything is configured in the `elemcoil` file
2. **Clean separation**: Molpro handles quantum chemistry, Julia handles specialized calculations
3. **Bidirectional communication**: Variables and data can flow both ways between Molpro and Julia
4. **Logging**: All Julia output is captured in `elemcoil.log` for debugging
5. **Flexible integration**: Commented sections allow for orbital modifications when needed

### Use Cases

- **Post-processing**: Analyze Molpro results with Julia's rich ecosystem
- **Method development**: Implement new electronic structure methods in Julia
- **Data analysis**: Use Julia's plotting and statistical capabilities on quantum chemistry data
- **Orbital manipulation**: Modify or transform molecular orbitals using Julia algorithms
- **Property calculations**: Compute additional molecular properties not available in Molpro

This interface makes it easy to leverage Julia's computational capabilities within existing 
Molpro workflows while maintaining clean, readable code.
"""
module MolproInterface

using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets

export is_matrop_file
export read_matrop_matrix, import_overlap, import_orbitals

include("molproXml.jl")

"""
    MOLPRO2LIBCINT_PERMUTATION

  Permutation of the atomic orbitals from the Molpro to the libcint order.
"""
const MOLPRO2LIBCINT_PERMUTATION = [        # Molpro order:
  [1],                                      # s 
  [1,2,3],                                  # p x,y,z
  [2,5,1,3,4],                              # d  0, -2, +1, +2, -1 (z^2, xy, xz, x^2-y^2, yz)
  [6,5,2,3,1,7,4],                          # f +1, -1,  0, +3, -2, -3, +2
  [7,9,2,5,1,3,6,8,4],                      # g  0, -2, +1, +4, -1, +2, -4, +3, -3
  [8,5,6,11,2,9,1,3,4,7,10],                # h +1, -1, +2, +3, -4, -3, +4, -5,  0, +5, -2
  [7,5,9,11,2,12,10,13,6,8,4,3,1]           # i +6, -2, +5, +4, -5, +2, -6, +3, -4,  0, -3, -1, +1
      ]

"""
    ao_permutation(EC::ECInfo)

  Return the permutation of the atomic orbitals from the Molpro to the libcint order 
  such that `μ(molpro)[ao_permutation(EC)] = μ(libcint)`.
"""
function ao_permutation(EC::ECInfo)
  basisset = generate_basis(EC, "ao")
  @assert !is_cartesian(basisset) "Only spherical basis sets are supported in import."
  return ao_order2internal(basisset, MOLPRO2LIBCINT_PERMUTATION)
end

"""
    skip_comment_lines(f::IOStream)

  Skip lines which do not start with a number or a minus.
"""
function skip_comment_lines(f::IOStream)
  line = position(f)
  while !occursin(r"^\s*[\d-]", readline(f))
    line = position(f)
  end
  # go back one line
  seek(f, line)
end

"""
    read_numbers_in_line(f::IOStream)

  Read a line from a file and return the numbers in it.
"""
function read_numbers_in_line(f::IOStream)
  if eof(f)
    return [], false
  end
  line = readline(f)
  if occursin(r"^\s*[\d-]", line)
    return [parse(Float64, x) for x in split(line, [' ',','], keepempty=false)], true
  else
    return [], false
  end
end

"""
    is_matrop_file(filename::AbstractString)

  Check if a file is a Molpro matrop file and return the type of the matrix.
"""
function is_matrop_file(filename::AbstractString)
  type = :NONE
  ismatrop = false
  open(filename) do f
    line = readline(f)
    ismatrop = occursin(r"^\s*BEGIN_DATA,", line)
    if ismatrop
      line = readline(f)
      # check type: "# MATRIX ORB1               ORBITALS
      info = split(line)
      if length(info) < 3 || info[1] != "#" || info[2] != "MATRIX"
        ismatrop = false
      else
        if info[4] == "ORBITALS"
          type = :ORBITALS
        elseif info[4] == "S"
          type = :OVERLAP
        end
      end
    end
  end
  return ismatrop, type
end

"""
    read_matrop_matrix(filename::AbstractString)

  Read a square matrix from a Molpro matrop file.
"""
function read_matrop_matrix(filename::AbstractString)
  vec = Float64[]
  open(filename) do f
    skip_comment_lines(f)
    # read matrix
    while true
      tmpvec, success = read_numbers_in_line(f)
      if !success
        break
      end
      append!(vec, tmpvec)
    end    
  end
  len = length(vec)
  dim = round(Int,sqrt(len))
  if dim^2 != len
    error("Matrix is not square! Length: $len, dimension: $dim")
  end
  return reshape(vec, dim, dim)' 
end

"""
    import_overlap(EC::ECInfo, filename::AbstractString)

  Import the overlap matrix from a Molpro matrop file.
"""
function import_overlap(EC::ECInfo, filename::AbstractString)
  println("Importing Molpro overlap from $filename")
  order = ao_permutation(EC)
  mat = read_matrop_matrix(filename)
  if size(mat) != (length(order), length(order))
    println("AO basis length: $(length(order))")
    error("Overlap matrix has wrong size: $(size(mat))")
  end
  return mat[order, order]
end

"""
    import_orbitals(EC::ECInfo, filename::AbstractString)

  Import an orbital coefficient matrix from a Molpro matrop file.
"""
function import_orbitals(EC::ECInfo, filename::AbstractString)
  println("Importing Molpro orbitals from $filename")
  order = ao_permutation(EC)
  mat = read_matrop_matrix(filename)
  if size(mat) != (length(order), length(order))
    println("AO basis length: $(length(order))")
    error("Orbital matrix has wrong size: $(size(mat))")
  end
  return mat[order, :]
end

end # module