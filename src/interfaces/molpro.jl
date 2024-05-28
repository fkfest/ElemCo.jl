"""
MolproInterface

This module provides an interface to Molpro to read and write orbitals and other data.
"""
module MolproInterface

using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets

export is_matrop_file
export read_matrop_matrix, import_overlap, import_orbitals

"""
    MOLPRO2LIBCINT_PERMUTATION

  Permutation of the atomic orbitals from the Molpro to the libcint order.
"""
const MOLPRO2LIBCINT_PERMUTATION = [        # Molpro order:
  [1],                                      # s 
  [1,2,3],                                  # p x,y,z
  [2,5,1,3,4],                              # d z^2, xy, xz, x^2-y^2, yz
  [6,5,2,3,1,7,4],                          # f 
  [7,9,2,5,1,3,6,8,4],                      # g
  [8,5,6,11,2,9,1,3,4,7,10],                # h
  [7,5,9,11,2,12,10,13,6,8,4,3,1]           # i
      ]

"""
    ao_permutation(EC::ECInfo)

  Return the permutation of the atomic orbitals from the Molpro to the libcint order 
  such that `μ(molpro)[ao_permutation(EC)] = μ(libcint)`.
"""
function ao_permutation(EC::ECInfo)
  basisset = generate_basis(EC, "ao")
  @assert !is_cartesian(basisset) "Only spherical basis sets are supported in import."
  order = Int[]
  for ash in basisset
    for ish = 1:n_subshells(ash)
      append!(order, MOLPRO2LIBCINT_PERMUTATION[ash.l+1] .+ length(order))
    end
  end
  return order
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