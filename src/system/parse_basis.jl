const BASIS_LIB = joinpath(@__DIR__, "..", "..", "lib", "basis_sets")

"""
    parse_basis(basis_name::String, atom::ACentre; fallback="", split_ashells=true) 

  Search and parse the basis set for a given atom.
  If `fallback` is non-empty, use as a fallback basis set.
  If `split_ashells` is true, split independent angular shells (important for efficiency).

  Return a list of angular shells [`AngularShell`](@ref).
"""
function parse_basis(basis_name::String, atom::ACentre; fallback="", split_ashells=true)
  # parse diffuse and steep components
  basis_name, add_diffuse, add_steep = parse_diffuse_steep(basis_name)
  if startswith(basis_name, "{")
    # the basis block is given explicitly, parse basis set from string
    return parse_basis_block(strip(basis_name, ['{','}'] ) , atom; add_diffuse, add_steep, split_ashells)
  else
    basisfile = basis_file(basis_name)
    if basisfile == ""
      if fallback != ""
        println(atomic_centre_label(atom),": Basis set $basis_name not found, using $fallback as a fallback.")
        basisfile = basis_file(fallback)
      else
        suggestions = suggest_basis_sets(basis_name)
        if !isempty(suggestions)
          suggestion_str = join(suggestions, ", ")
          error("Basis set $basis_name not found! Did you mean: $suggestion_str?")
        else
          error("Basis set $basis_name not found!")
        end
      end
    end
    basisblock = read_basis_block(basisfile, atom; fallback)
  end
  return parse_basis_block(basisblock, atom; add_diffuse, add_steep, split_ashells)
end

"""
    parse_diffuse_steep(basis_name::AbstractString)

  Parse the diffuse and steep components from the basis name in the format 
  `+<N>diffuse` or `+diffuse` or `+<N>steep` or `+steep`.

  Returns the modified basis name and the number of diffuse and steep functions to add.
"""
function parse_diffuse_steep(basis_name::AbstractString)
  # remove possible -jkfit/-mpfit/-rifit suffixes (will be added back later)
  if occursin(r"-(jkfit|mpfit|rifit)$", basis_name)
    suffix = "-$(match(r"-(jkfit|mpfit|rifit)$", basis_name).captures[1])"
    basis_name = replace(basis_name, r"-(jkfit|mpfit|rifit)$" => "")
  else
    suffix = ""
  end
  add_diffuse = 0
  add_steep = 0
  while occursin(r"\+\d*(diffuse|steep)$", basis_name)
    if occursin(r"\+\d+diffuse$", basis_name)
      add_diffuse += parse(Int, match(basis_name, r"\+(\d+)diffuse$").captures[1])
      # remove +<N>diffuse from the basis name
      basis_name = replace(basis_name, r"\+\d+diffuse$" => "")
    elseif occursin(r"\+diffuse$", basis_name)
      add_diffuse += 1
      # remove +diffuse from the basis name
      basis_name = replace(basis_name, r"\+diffuse$" => "")
    elseif occursin(r"\+\d+steep$", basis_name)
      add_steep += parse(Int, match(basis_name, r"\+(\d+)steep$").captures[1])
      # remove +<N>steep from the basis name
      basis_name = replace(basis_name, r"\+\d+steep$" => "")
    elseif occursin(r"\+steep$", basis_name)
      add_steep += 1
      # remove +steep from the basis name
      basis_name = replace(basis_name, r"\+steep$" => "")
    end
  end
  # add removed jkfit/mpfit/rifit suffixes back and return
  return basis_name*suffix, add_diffuse, add_steep
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
    # Basis set not found
    return ""
  end
  filename = "$mainname.$version.mpro"
  if !isfile(filename)
    suggestions = suggest_basis_sets(basis_name)
    if !isempty(suggestions)
      suggestion_str = join(suggestions, ", ")
      error("Basis set $basis_name version $version not found! Did you mean: $suggestion_str?")
    else
      error("Basis set $basis_name version $version not found!")
    end
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
    read_basis_block(basisfile::AbstractString, atom::ACentre; fallback="") 

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

  If the basis block is not found, the function will attempt to read the fallback basis file.
"""
function read_basis_block(basisfile::AbstractString, atom::ACentre; fallback="")
  elem = lowercase(element_fullname(atom))
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
    if fallback != ""
      println("Basis block for $elem not found in $(basisfile). Using fallback basis set $fallback.")
      basisfile = basis_file(fallback)
      if basisfile == ""
        error("Fallback basis set $fallback not found!")
      end
      return read_basis_block(basisfile, atom; fallback="")
    else
      error("Basis block for $elem not found in $(basisfile)!")
    end
  end
  return basisblock
end

"""
    parse_basis_block(basis_block::AbstractString, atom::ACentre; add_diffuse=0, add_steep=0, split=true) 

  Parse the basis block for a given atom.

  Return a list of angular shells [`AngularShell`](@ref).
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

`add_diffuse` and `add_steep` are the number of diffuse and steep even-tempered functions to add.
If `split_ashells` is true, independent angular shells will be split (important for efficiency).
"""
function parse_basis_block(basis_block::AbstractString, atom::ACentre; add_diffuse=0, add_steep=0, split_ashells=true)
  basisblock = lowercase(basis_block)
  elem = lowercase(element_LABEL(atom))
  # search for ` s, $elem , 13...`
  reg_exp = Regex("^\\s*[$SUBSHELLS_NAMES]\\s*,\\s*$elem\\s*,")
  reg_con = Regex("^\\s*c,\\s*")
  ashells = AngularShell[]
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
      push!(ashells, generate_angularshell(elem, parse_exponents(line; add_diffuse, add_steep)...))
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
  # add uncontracted exponents as uncontracted subshells and split angular shells if necessary
  ashells_split = AngularShell[]
  for ashell in ashells
    add_uncontracted!(ashell)
    if split_ashells
      append!(ashells_split, split_angular_shell(ashell))
    else
      push!(ashells_split, ashell)
    end
  end
  return ashells_split
end

"""
    parse_exponents(expline::AbstractString; add_diffuse=0, add_steep=0)

  Parse exponents from a line in the basis block.

  Return the angular momentum and exponents as a tuple.
  The line is in the Molpro format:
  `s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000`
  where `s` is the angular momentum, `H` is the element symbol,
  and the rest are the exponents.

  The `add_diffuse` and `add_steep` arguments can be used to add diffuse and steep functions.
"""
function parse_exponents(expline::AbstractString; add_diffuse=0, add_steep=0)
  # parse exponents
  exponents = strip.(split(expline, ","))
  lval = SUBSHELL2L[exponents[1][1]]
  # remove angular momentum and element symbol and convert to Float64
  exponents = parse.(Float64, exponents[3:end])
  add_steep_exponent!(exponents, add_steep)
  add_diffuse_exponent!(exponents, add_diffuse)
  return lval, exponents
end

"""
    add_steep_exponent!(exponents, nexp)

  Add `nexp` steep even-tempered exponents to the list of exponents.

  Each new exponent is generated as `e = e1^2/e2`, where `e1` is the steepest and `e2` the second
  steepest exponent (the list of exponents is not necessarily ordered). 
  If only one exponent is in the list, the new exponent is set to `e1*2.5`.
  The new exponents are added to the end of the list.
"""
function add_steep_exponent!(exponents, nexp)
  for iexp in 1:nexp
    if length(exponents) == 1
      # if only one exponent, add steep exponent as e1*2.5
      push!(exponents, exponents[1] * 2.5)
    else
      # find the steepest and second steepest exponents
      i1, i2 = argmaxN(exponents, 2)
      e1 = exponents[i1]
      e2 = exponents[i2]
      # add new steep exponent as e1^2/e2
      push!(exponents, e1^2 / e2)
    end
  end
  return exponents
end

"""
    add_diffuse_exponent!(exponents, nexp)

  Add `nexp` diffuse even-tempered exponents to the list of exponents.

  Each new exponent is generated as `e = e1^2/e2`, where `e1` is the most diffuse and `e2` the second
  most diffuse exponent (the list of exponents is not necessarily ordered). 
  If only one exponent is in the list, the new exponent is set to `e1/2.5`.
  The new exponents are added to the end of the list.
"""
function add_diffuse_exponent!(exponents, nexp)
  for iexp in 1:nexp
    if length(exponents) == 1
      # if only one exponent, add diffuse exponent as e1/2.5
      push!(exponents, exponents[1] / 2.5)
    else
      # find the most diffuse and second most diffuse exponents
      i1, i2 = argmaxN(exponents, 2; by=(x -> -x))
      e1 = exponents[i1]
      e2 = exponents[i2]
      # add new diffuse exponent as e1^2/e2
      push!(exponents, e1^2 / e2)
    end
  end
  return exponents
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
  start, stop = parse.(Int, split(contraction[2], "."))
  exprange = start:stop
  # remove contraction and exponent range and convert to Float64
  contraction = parse.(Float64, contraction[3:end])
  if length(contraction) != length(exprange)
    println("Problem in contraction line $conline")
    error("Number of contraction coefficients does not match the number of exponents in the range!")
  end
  return exprange, contraction
end

"""
    add_uncontracted!(ashell::AngularShell)

  Exponents not covered by contractions are added as uncontracted subshells.
"""
function add_uncontracted!(ashell::AngularShell)
  # find uncontracted exponents
  used_exps = zeros(Bool, length(ashell.exponents))
  for sh in ashell.subshells
    used_exps[sh.exprange] .= true
  end
  # add uncontracted subshells
  for (i, contracted) in enumerate(used_exps)
    if !contracted
      add_subshell!(ashell, i:i, [1.0])
    end
  end
end

"""
    split_angular_shell(ashell::AngularShell)

  If the ranges of exponents do not overlap, split the angular shell
  into separate angular shells for each subshell.
  The shells are kept together only if one is a subset of the other.
"""
function split_angular_shell(ashell::AngularShell)
  ers = [sh.exprange for sh in ashell.subshells]
  # intersection matrix for ranges of exponents (true if one is a subset of the other)
  imat = [length(intersect(r1, r2)) == min(length(r1),length(r2)) for r1 in ers, r2 in ers]
  # find ranges of block-diagonal blocks in the intersection matrix
  blocks = UnitRange{Int}[]
  start = 1
  for i in 1:length(ers)
    if !any(imat[start:i,i+1:end])
      push!(blocks, start:i)
      start = i+1
    end
  end
  # split the angular shell
  ashells = AngularShell[]
  for block in blocks
    # total exponent range for this block
    totexprange = minimum(ers[block]).start:maximum(ers[block]).stop
    push!(ashells, generate_angularshell(ashell.element, ashell.l, ashell.exponents[totexprange] ))
    for i in block
      exprange = subspace_in_space(ashell.subshells[i].exprange, totexprange)
      add_subshell!(last(ashells), exprange, ashell.subshells[i].coefs)
    end
  end
  return ashells
end

"""
    get_available_elements4basis(basisname)

  Return a list of available elements for the specified basis set.
"""
function get_available_elements4basis(basisname)
  basisname, add_diffuse, add_steep = parse_diffuse_steep(basisname)
  basisfile = basis_file(basisname)
  @assert basisfile != "" "Basis set $basisname not found!"
  elements = String[]
  # search for ` s, $elem , 13...`
  reg_exp = Regex("^\\s*s\\s*,\\s*([^,]+)\\s*,")
  open(basisfile) do io
    for line in eachline(io)
      if occursin(reg_exp, line)
        push!(elements, strip(match(reg_exp, line).captures[1]))
      end
    end
  end
  return elements
end