const BASIS_LIB = joinpath(@__DIR__, "..", "..", "lib", "basis_sets")

"""
    parse_basis(basis_name::String, atom::Atom; cartesian=false) 

  Search and parse the basis set for a given atom.

  Return a list of subshells `AbstractSubShell`.
"""
function parse_basis(basis_name::String, atom::Atom; cartesian=false)
  basisfile = basis_file(basis_name)
  basisblock = read_basis_block(basisfile, atom)
  return parse_basis_block(basisblock, atom; cartesian)
end

"""
    basis_file(basis_name::AbstractString) 

  Return the full path to the basis set file.
"""
function basis_file(basis_name::AbstractString)
  # expand basis names
  basis_name, version = full_basis_name(lowercase(basis_name))
  # replace [* => _st_, mpfit => rifit] 
  basis_name = replace(basis_name, "*" => "_st_", "mpfit" => "rifit")
  mainname = joinpath(BASIS_LIB, "mpro", basis_name)
  if version < 0
    for ver in 2:-1:0
      if isfile("$mainname.$ver.mpro")
        version = ver
        break
      end
    end
  end 
  if version < 0
    error("Basis set $basis_name not found!")
  end
  filename = "$mainname.$version.mpro"
  if !isfile(filename)
    error("Basis set $basis_name version $version not found!")
  end
  return filename
end

"""
    full_basis_name(basis_name::AbstractString) 

  Return the full basis name and version number 
  (if given as `*.v[0-2]`, otherwise `-1` is returned).
  
  I.e,
  - `[a][wc/c]vXz*` -> `[aug-]cc-p[wc/c]vXz*`
  - `svp*` -> `def2-svp*`
  - `[tq]zvp*` -> `def2-[tq]zvp*`
  Additionally check for version number (e.g., `vdz.v2`)
"""
function full_basis_name(basis_name::AbstractString)
  # check for version number
  version = -1
  if occursin(r"\.v[0-2]$", basis_name)
    version = parse(Int, last(basis_name))
    basis_name = basis_name[1:end-3]
  end
  if occursin(r"^[a]?w?c?v[dtq5-9]z", basis_name)
    # expand [a][wc/c]vNz* basis names
    basis_name = basis_name[1] == 'a' ? "aug-cc-p$(basis_name[2:end])" : "cc-p$basis_name"
  elseif occursin(r"^[dtq]zvp", basis_name)
    # expand def2 basis names
    basis_name = "def2-$basis_name"
  elseif occursin(r"^svp", basis_name)
    # expand def2-svp basis names
    basis_name = "def2-$basis_name"
  end
  return basis_name, version
end

"""
    read_basis_block(basisfile::AbstractString, atom::Atom) 

  Read the basis block for a given atom.

  The basis library is in the Molpro format:
  - `!` comments
  - basis block starts with `! <elementname>  ....`
  - basis block ends with `!` or `}`
  - basis block contains:
  - `s,p,d,f,g,h` angular momentum
  - `c, <from>.<to>` contraction coefficients for primitives

  Example cc-pVDZ for H atom:
```
!
! hydrogen             (4s,1p) -> [2s,1p]
s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000
c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400
c, 4.4, 1.0000000
p, H , 0.7270000
c, 1.1, 1.0000000
!
```
"""
function read_basis_block(basisfile::AbstractString, atom::Atom)
  elem = lowercase(element_name(atom))
  # search for `! $elem  ....`
  reg_start = Regex("^!\\s$elem\\s+")
  reg_end = Regex("^\\s*[!}]\\s*")
  basisblock::String = ""
  open(basisfile) do f
    elemfound = false
    for line::String in eachline(f)
      if elemfound
        if occursin(reg_end, line)
          break
        else
          basisblock *= line * "\n"
        end
      else
        elemfound = occursin(reg_start, line)
      end
    end
  end
  if isempty(basisblock)
    error("Basis block for $elem not found in $(basisfile)!")
  end
  return basisblock
end

"""
    parse_basis_block(basisblock::AbstractString, atom::Atom; cartesian=false) 

  Parse the basis block for a given atom.

  Return a list of angular shells [`AbstractAngularShell`](@ref).
  The basis block is in the Molpro format:
  - `!` comments
  - `s,p,d,f,g,h` angular momentum
  - `c, <from>.<to>` contraction coefficients for primitives

  Example cc-pVDZ for H atom:
```
s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000
c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400
c, 4.4, 1.0000000
p, H , 0.7270000
c, 1.1, 1.0000000
```

  For generally-contracted basis sets (like the one above), one angular shell
  is created for each angular momentum type `s,p,d,f,g,h` with the corresponding
  exponents and contraction coefficients. For other basis sets, like the def2-SVP,
  each contraction is a separate angular shell:
```
! hydrogen             (4s,1p) -> [2s,1p]
s, H , 13.0107010, 1.9622572, 0.44453796, 0.12194962
c, 1.3, 0.19682158E-01, 0.13796524, 0.47831935
c, 4.4, 1.0000000
p, H , 0.8000000
c, 1.1, 1.0000000
```
"""
function parse_basis_block(basisblock::AbstractString, atom::Atom; cartesian=false)
  elem = element_SYMBOL(atom)
  # search for ` s, $elem , 13...`
  reg_exp = Regex("^\\s*[$SUBSHELLS_NAMES]\\s*,\\s*$elem\\s*,")
  reg_con = Regex("^\\s*c,\\s*")
  ashells = AbstractAngularShell[] # cartesian ? CartesianAngularShell[] : SphericalAngularShell[]
  for line in split(basisblock, "\n")
    #remove comments ` abc !...` -> `abc`
    line = strip(replace(line, r"!.*" => ""))
    #and empty lines
    if isempty(line)
      continue
    end
    expline = occursin(reg_exp, line)
    if expline
      # parse exponents
      push!(ashells, generate_angularshell(elem, parse_exponents(line)...; cartesian))
    else
      conline = occursin(reg_con, line)
      if conline
        if isempty(ashells)
          println("Problem in basis block $basisblock")
          error("Contraction line before exponents line!")
        end
        # parse contraction coefficients
        exprange,contraction = parse_contraction(line)
        if exprange.stop > length(last(ashells).exponents)
          println("Problem in basis block $basisblock")
          error("Exponent range exceeds the number of exponents!")
        end
        # add subshell
        add_subshell!(last(ashells), exprange, contraction)
      end
    end
  end
  # split angular shells if necessary
  ashells_split = AbstractAngularShell[]
  for ashell in ashells
    append!(ashells_split, split_angular_shell(ashell; cartesian))
  end
  return ashells_split
end

"""
    parse_exponents(expline::AbstractString)

  Parse exponents from a line in the basis block.

  Return the angular momentum and exponents as a tuple.
  The line is in the Molpro format:
  `s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000`
  where `s` is the angular momentum, `H` is the element symbol,
  and the rest are the exponents.
"""
function parse_exponents(expline::AbstractString)
  # parse exponents
  exponents = strip.(split(expline, ","))
  lval = SUBSHELL2L[exponents[1][1]]
  # remove angular momentum and element symbol and convert to Float64
  exponents = parse.(Float64, exponents[3:end])
  return lval, exponents
end

"""
    parse_contraction(conline::AbstractString)

  Parse contraction coefficients from a line in the basis block.

  Return the range of exponents and the contraction coefficients as a tuple.
  The line is in the Molpro format:
  `c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400`
  where `c` is the contraction, `1.4` is the exponent range,
  and the rest are the coefficients.
"""
function parse_contraction(conline::AbstractString)
  # parse contraction coefficients
  contraction = strip.(split(conline, ","))
  # parse exponent range
  exprange = range(parse.(Int,split(contraction[2], "."))...)
  # remove contraction and exponent range and convert to Float64
  contraction = parse.(Float64, contraction[3:end])
  if length(contraction) != length(exprange)
    println("Problem in contraction line $conline")
    error("Number of contraction coefficients does not match the number of exponents in the range!")
  end
  return exprange, contraction
end

"""
    split_angular_shell(ashell::AbstractAngularShell; cartesian=false)

  If the ranges of exponents do not overlap, split the angular shell
  into separate angular shells for each subshell.
  The shells are kept together only if one is a subset of the other.
"""
function split_angular_shell(ashell::AbstractAngularShell; cartesian=false)
  ers = [sh.exprange for sh in ashell.subshells]
  # intersection matrix for ranges of exponents (true if one is a subset of the other)
  imat = [length(intersect(r1, r2)) == min(length(r1),length(r2)) for r1 in ers, r2 in ers]
  # find ranges of block-diagonal blocks in the intersection matrix
  blocks = UnitRange[]
  start = 1
  for i in 1:length(ers)
    if !any(imat[start:i,i+1:end])
      push!(blocks, start:i)
      start = i+1
    end
  end
  # split the angular shell
  ashells = AbstractAngularShell[]
  for block in blocks
    # total exponent range for this block
    totexprange = minimum(ers[block]).start:maximum(ers[block]).stop
    push!(ashells, generate_angularshell(ashell.element, ashell.l, ashell.exponents[totexprange]; cartesian))
    for i in block
      exprange = subspace_in_space(ashell.subshells[i].exprange, totexprange)
      add_subshell!(last(ashells), exprange, ashell.subshells[i].coefs)
    end
  end
  return ashells
end