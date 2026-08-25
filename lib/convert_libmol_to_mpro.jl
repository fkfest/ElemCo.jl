#!/usr/bin/env julia
"""
Convert Molpro library format (.libmol) to Molpro user format (.mpro).

Usage:
    julia convert_libmol_to_mpro.jl input.libmol [output_dir]

Converts all basis set variants found in the input file to separate .mpro files.
For example, minao.libmol containing MINAO, MINAO-PP, MINAO-EXTVAL, MINAO-EXTVAL-PP
will produce minao.0.mpro, minao-pp.0.mpro, minao-extval.0.mpro, minao-extval-pp.0.mpro
in the output directory.
"""

using Printf

struct ShellBlock
  element::String
  shell::String     # "s", "p", "d", "f", "g"
  nprim::Int
  ncontr::Int
  ranges::Vector{Tuple{Int,Int}}
  exponents::Vector{Float64}
  coefficients::Vector{Vector{Float64}}
end

const SHELL_ORDER = Dict("s" => 0, "p" => 1, "d" => 2, "f" => 3, "g" => 4, "h" => 5)

const ELEMENT_NAMES = Dict(
  "H" => "hydrogen", "He" => "helium", "Li" => "lithium", "Be" => "beryllium",
  "B" => "boron", "C" => "carbon", "N" => "nitrogen", "O" => "oxygen",
  "F" => "fluorine", "Ne" => "neon", "Na" => "sodium", "Mg" => "magnesium",
  "Al" => "aluminum", "Si" => "silicon", "P" => "phosphorus", "S" => "sulfur",
  "Cl" => "chlorine", "Ar" => "argon", "K" => "potassium", "Ca" => "calcium",
  "Sc" => "scandium", "Ti" => "titanium", "V" => "vanadium", "Cr" => "chromium",
  "Mn" => "manganese", "Fe" => "iron", "Co" => "cobalt", "Ni" => "nickel",
  "Cu" => "copper", "Zn" => "zinc", "Ga" => "gallium", "Ge" => "germanium",
  "As" => "arsenic", "Se" => "selenium", "Br" => "bromine", "Kr" => "krypton",
  "Rb" => "rubidium", "Sr" => "strontium", "Y" => "yttrium", "Zr" => "zirconium",
  "Nb" => "niobium", "Mo" => "molybdenum", "Tc" => "technetium", "Ru" => "ruthenium",
  "Rh" => "rhodium", "Pd" => "palladium", "Ag" => "silver", "Cd" => "cadmium",
  "In" => "indium", "Sn" => "tin", "Sb" => "antimony", "Te" => "tellurium",
  "I" => "iodine", "Xe" => "xenon", "Cs" => "cesium", "Ba" => "barium",
  "La" => "lanthanum", "Ce" => "cerium", "Pr" => "praseodymium", "Nd" => "neodymium",
  "Pm" => "promethium", "Sm" => "samarium", "Eu" => "europium", "Gd" => "gadolinium",
  "Tb" => "terbium", "Dy" => "dysprosium", "Ho" => "holmium", "Er" => "erbium",
  "Tm" => "thulium", "Yb" => "ytterbium", "Lu" => "lutetium", "Hf" => "hafnium",
  "Ta" => "tantalum", "W" => "tungsten", "Re" => "rhenium", "Os" => "osmium",
  "Ir" => "iridium", "Pt" => "platinum", "Au" => "gold", "Hg" => "mercury",
  "Tl" => "thallium", "Pb" => "lead", "Bi" => "bismuth", "Po" => "polonium",
  "At" => "astatine", "Rn" => "radon", "U" => "uranium",
)

"""Parse a contraction range string like "1.5" → (1, 5), "1.20" → (1, 20)."""
function parse_range_str(s::AbstractString)
  parts = split(s, ".")
  return (parse(Int, parts[1]), parse(Int, parts[2]))
end

"""
    parse_libmol(filename) → Dict{String, Vector{ShellBlock}}

Parse a .libmol file and return a dictionary mapping basis set names
to their shell blocks. Blocks belonging to multiple basis sets (e.g.,
"Sc d MINAO MINAO-EXTVAL") are included in each named set.
"""
function parse_libmol(filename::String)
  lines = readlines(filename)
  blocks = Dict{String, Vector{ShellBlock}}()

  i = 1
  while i <= length(lines)
    line = strip(lines[i])

    # Skip empty lines and comments
    if isempty(line) || startswith(line, "!")
      i += 1
      continue
    end

    # Header line: Element shell BASISNAME1 [BASISNAME2...] : nprim ncontr range1 ...
    # Must start with element symbol (1-2 chars, capital+optional lowercase)
    # followed by shell type (s, p, d, f, g)
    if !contains(line, ":")
      i += 1
      continue
    end

    m = match(r"^([A-Z][a-z]?)\s+([spdfgh])\s+(.+?)\s*:\s*(.+)$", line)
    if m === nothing
      i += 1
      continue
    end

    element = m.captures[1]
    shell = m.captures[2]
    basis_names = String.(split(m.captures[3]))
    data_parts = split(strip(m.captures[4]))

    nprim = parse(Int, data_parts[1])
    ncontr = parse(Int, data_parts[2])
    ranges = [parse_range_str(data_parts[2+j]) for j in 1:ncontr]

    # Total data values: nprim exponents + coefficients for each contraction
    ncoeffs_total = sum(r[2] - r[1] + 1 for r in ranges)
    ndata = nprim + ncoeffs_total

    # Read data values from subsequent lines (flat number stream)
    i += 1
    values = Float64[]
    while length(values) < ndata && i <= length(lines)
      dline = strip(lines[i])
      if isempty(dline) && isempty(values)
        # Skip blank line between header and data
        i += 1
        continue
      end
      if isempty(dline) || startswith(dline, "!")
        break
      end
      # Skip inline comments (e.g., "cgk: ...")
      tokens = split(dline)
      num = tryparse(Float64, tokens[1])
      if num === nothing
        i += 1
        continue
      end
      nums = parse.(Float64, tokens)
      append!(values, nums)
      i += 1
    end

    if length(values) != ndata
      error("$element $shell $(join(basis_names, " ")): expected $ndata values, got $(length(values))")
    end

    exponents = values[1:nprim]
    coefficients = Vector{Float64}[]
    offset = nprim
    for (start, stop) in ranges
      n = stop - start + 1
      push!(coefficients, values[offset+1:offset+n])
      offset += n
    end

    block = ShellBlock(element, shell, nprim, ncontr, ranges, exponents, coefficients)

    for name in basis_names
      if !haskey(blocks, name)
        blocks[name] = ShellBlock[]
      end
      push!(blocks[name], block)
    end
  end

  return blocks
end

"""Format a number with 10 significant digits."""
function fmt(x::Float64)
  @sprintf("%.10g", x)
end

"""Build a summary string like (5s,3p) -> [1s,2p] for an element's shells."""
function element_summary(shells::Vector{ShellBlock})
  prim_parts = String[]
  contr_parts = String[]
  for s in sort(shells, by=b->get(SHELL_ORDER, b.shell, 9))
    push!(prim_parts, "$(s.nprim)$(s.shell)")
    push!(contr_parts, "$(s.ncontr)$(s.shell)")
  end
  return "($(join(prim_parts, ","))) -> [$(join(contr_parts, ","))]"
end

"""
    write_mpro(filename, basis_name, blocks)

Write shell blocks to a .mpro file in Molpro user format.
"""
function write_mpro(filename::String, basis_name::String, blocks::Vector{ShellBlock})
  # Group blocks by element, preserving first-appearance order
  element_order = String[]
  element_blocks = Dict{String, Vector{ShellBlock}}()
  for block in blocks
    if !haskey(element_blocks, block.element)
      push!(element_order, block.element)
      element_blocks[block.element] = ShellBlock[]
    end
    push!(element_blocks[block.element], block)
  end

  open(filename, "w") do io
    println(io, "!----------------------------------------------------------------------")
    println(io, "! Converted from Molpro library format (libmol)")
    println(io, "!----------------------------------------------------------------------")
    println(io, "!   Basis set: $basis_name")
    println(io, "!----------------------------------------------------------------------")
    println(io)
    println(io, "spherical")
    println(io, "basis={")

    for elem in element_order
      shells = element_blocks[elem]
      summary = element_summary(shells)
      elem_upper = uppercase(elem)
      elem_name = get(ELEMENT_NAMES, elem, lowercase(elem))

      println(io, "!")
      println(io, "! $(rpad(elem_name, 20)) $summary")

      # Sort shells by angular momentum
      sort!(shells, by=b->get(SHELL_ORDER, b.shell, 9))

      for shell in shells
        # Exponent line: s, H , exp1, exp2, ...
        exp_strs = [fmt(e) for e in shell.exponents]
        println(io, "$(shell.shell), $elem_upper , $(join(exp_strs, ", "))")

        # Contraction lines: c, start.end, coeff1, coeff2, ...
        for (j, ((start, stop), coeffs)) in enumerate(zip(shell.ranges, shell.coefficients))
          coeff_strs = [fmt(c) for c in coeffs]
          println(io, "c, $start.$stop, $(join(coeff_strs, ", "))")
        end
      end
    end

    println(io, "}")
  end
end

function main()
  if length(ARGS) < 1
    println("Usage: julia convert_libmol_to_mpro.jl input.libmol [output_dir]")
    return
  end

  input_file = ARGS[1]
  output_dir = length(ARGS) >= 2 ? ARGS[2] : dirname(input_file)

  println("Parsing $input_file...")
  blocks = parse_libmol(input_file)

  for (basis_name, shell_blocks) in sort(collect(blocks), by=first)
    # Convert basis name to filename: MINAO → minao, MINAO-PP → minao-pp
    filename = lowercase(basis_name) * ".0.mpro"
    filepath = joinpath(output_dir, filename)

    elements = unique(b.element for b in shell_blocks)

    println("  $basis_name → $filename ($(length(shell_blocks)) shells, $(length(elements)) elements)")
    write_mpro(filepath, basis_name, shell_blocks)
  end

  println("Done!")
end

main()
