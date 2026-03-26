#!/usr/bin/env julia
"""
    generate_fields.jl

Parse the TREXIO specification file `trex.org` and generate Julia source code
defining the standard `TrexioField` arrays.  The output file can be included
into `TREXIO.jl` via `include(...)`, replacing the hand-written standard field
definitions.

Usage:
    julia generate_fields.jl [trex.org] [output.jl]

Defaults:
    trex.org  →  <script_dir>/trex.org
    output.jl →  <script_dir>/src/trexio_standard_fields.jl
"""

# ---------------------------------------------------------------------------
# Type mapping from trex.org notation to Julia
# ---------------------------------------------------------------------------

"""Map a trex.org type string to (julia_type, is_sparse)."""
function map_trexio_type(raw::AbstractString)
  t = strip(raw)
  is_sparse = occursin("sparse", t)
  # strip qualifiers: "sparse", "buffered", "readonly", "special"
  base = replace(t, r"\s*(sparse|buffered|readonly|special)\s*" => "")
  base = strip(base)
  jtype = if base in ("dim", "index", "int")
    "Int"
  elseif base == "float"
    "Float64"
  elseif base == "str"
    "String"
  else
    error("Unknown trex.org type: \"$raw\"")
  end
  return jtype, is_sparse
end

"""
Parse column-major dimension string from trex.org, e.g. `~(3, nucleus.num)~`,
returning a `Vector{String}` of dimension components.
Scalar (empty) returns `String[]`.
"""
function parse_dimensions(raw::AbstractString)
  s = strip(raw)
  # remove surrounding tildes
  s = replace(s, "~" => "")
  s = strip(s)
  isempty(s) && return String[]
  # expect parenthesised list  (d1, d2, ...)
  m = match(r"^\((.+)\)$", s)
  isnothing(m) && return String[]
  inner = m.captures[1]
  parts = split(inner, ",")
  return [strip(String(p)) for p in parts]
end

# ---------------------------------------------------------------------------
# Org-mode parser
# ---------------------------------------------------------------------------

struct OrgField
  variable::String
  juliatype::String
  sparse::Bool
  dimensions::Vector{String}
  description::String
end

"""
Parse `trex.org` and return a `Vector{Pair{String, Vector{OrgField}}}` where
each pair is `group_name => fields`.  The order of groups and fields is
preserved.
"""
function parse_trex_org(filepath::String)
  lines = readlines(filepath)
  groups = Pair{String, Vector{OrgField}}[]

  current_name = ""          # set by #+NAME:
  in_table = false
  header_seen = false
  fields = OrgField[]

  for line in lines
    stripped = strip(line)

    # Detect #+NAME: tag  →  next table belongs to this group
    m = match(r"^#\+NAME:\s*(\S+)"i, stripped)
    if !isnothing(m)
      current_name = m.captures[1]
      in_table = false
      header_seen = false
      fields = OrgField[]
      continue
    end

    # Detect table rows (lines starting with |)
    if startswith(stripped, "|")
      if isempty(current_name)
        # table without a preceding #+NAME: — skip
        continue
      end

      # separator row  |---+---+...|
      if occursin(r"^\|[-+]+\|$", stripped)
        if header_seen
          # end of header separator — data rows follow
        end
        continue
      end

      # split columns by | — keep positional indexing (don't filter empties)
      cols = split(stripped, "|")
      # cols[1] is before first |, cols[end] after last |
      # actual table columns are at indices 2 .. end-1
      ncols = length(cols) - 2   # number of actual table columns

      if !header_seen
        # This is the header row — skip it but mark header as seen
        if ncols >= 5
          header_seen = true
        end
        continue
      end

      # Data row — expect ≥5 columns:
      #   col[2]: variable, col[3]: type, col[4]: row-major dims, col[5]: col-major dims, col[6]: description
      ncols < 5 && continue

      varname = replace(strip(cols[2]), "~" => "")
      rawtype = replace(strip(cols[3]), "~" => "")
      # Column 5 (index) is column-major dimensions
      coldims = strip(cols[5])
      desc = strip(cols[6])
      # Clean up tildes in dims and description
      coldims = replace(coldims, "~" => "")
      desc = replace(desc, "~" => "")
      # Clean up any remaining leading/trailing whitespace or formatting
      desc = strip(desc)

      jtype, is_sparse = map_trexio_type(rawtype)
      dims = parse_dimensions(coldims)

      push!(fields, OrgField(varname, jtype, is_sparse, dims, desc))

      in_table = true
      continue
    end

    # If we were in a table and hit a non-table line, flush
    if in_table && !isempty(current_name)
      push!(groups, current_name => copy(fields))
      current_name = ""
      in_table = false
      header_seen = false
      fields = OrgField[]
    end
  end

  # Flush last table if file ends inside one
  if in_table && !isempty(current_name) && !isempty(fields)
    push!(groups, current_name => copy(fields))
  end

  return groups
end

# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

# Section header comments matching the structure in TREXIO.jl
const GROUP_COMMENTS = Dict(
  "metadata"    => "# 1. Metadata fields (stored in metadata group at root level)",
  "nucleus"     => "# 2.1 Nucleus (nucleus group)",
  "cell"        => "# 2.2 Cell (cell group)",
  "pbc"         => "# 2.3 Periodic boundary calculations (pbc group)",
  "electron"    => "# 2.4 Electron (electron group)",
  "state"       => "# 2.5 Ground or excited states (state group)",
  "basis"       => "# 3.1 Basis set (basis group)",
  "ecp"         => "# 3.2 Effective core potentials (ecp group)",
  "grid"        => "# 3.3 Numerical integration grid (grid group)",
  "ao"          => "# 4.1 Atomic orbitals (ao group)",
  "ao_1e_int"   => "# 4.1.1 One-electron integrals (ao_1e_int group)",
  "ao_2e_int"   => "# 4.1.2 Two-electron integrals (ao_2e_int group)",
  "mo"          => "# 4.2 Molecular orbitals (mo group)",
  "mo_1e_int"   => "# 4.2.1 One-electron integrals (mo_1e_int group)",
  "mo_2e_int"   => "# 4.2.2 Two-electron integrals (mo_2e_int group)",
  "determinant" => "# 5.1 Slater determinants (determinant group)",
  "csf"         => "# 5.2 Configuration state functions (csf group)",
  "amplitude"   => "# 5.3 Amplitudes (amplitude group)",
  "rdm"         => "# 5.4 Reduced density matrices (rdm group)",
  "jastrow"     => "# 6.1 Jastrow factor (jastrow group)",
  "qmc"         => "# 7. Quantum Monte Carlo data (qmc group)",
)

"""Format a dimensions vector as Julia source code."""
function format_dims(dims::Vector{String})
  isempty(dims) && return "SCALAR"
  # Check if all elements are the same and more than three → use fill(...)
  if length(dims) > 3 && allequal(dims)
    return "fill(\"$(dims[1])\", $(length(dims)))"
  end
  return "[" * join(["\"$d\"" for d in dims], ", ") * "]"
end

"""Escape a string for use inside a Julia string literal."""
function escape_description(s::String)
  s = replace(s, "\\" => "\\\\")
  s = replace(s, "\"" => "\\\"")
  s = replace(s, "\$" => "\\\$")
  return s
end

"""Generate the Julia source file with all standard TrexioField definitions."""
function generate_fields_file(groups::Vector{Pair{String, Vector{OrgField}}}, output::String)
  io = IOBuffer()

  println(io, "#")
  println(io, "# Standard TREXIO field definitions — AUTO-GENERATED from trex.org")
  println(io, "# DO NOT EDIT MANUALLY.  Re-generate with:  julia generate_fields.jl")
  println(io, "#")
  println(io, "# Source: https://trex-coe.github.io/trexio/trex.html")
  println(io, "#")
  println(io)

  const_names = String[]

  for (group, fields) in groups
    comment = get(GROUP_COMMENTS, group, "# $(group) group")
    println(io, comment)
    constname = "TREXIO_$(uppercase(group))_FIELDS"
    push!(const_names, constname)
    println(io, "const $constname = [")
    for (i, f) in enumerate(fields)
      dims_str = format_dims(f.dimensions)
      desc_str = escape_description(f.description)
      sparse_str = f.sparse ? ", sparse=true" : ""
      entry = "    TrexioField(\"$(group)\", \"$(f.variable)\", $(f.juliatype), $(dims_str), \"$(desc_str)\"$(sparse_str))"
      if i < length(fields)
        entry *= ","
      else
        entry *= ","
      end
      println(io, entry)
    end
    println(io, "]")
    println(io)
  end

  # Generate the combined constant
  println(io, "# Combine all standard field definitions")
  println(io, "const STANDARD_TREXIO_FIELDS = vcat(")
  for (i, name) in enumerate(const_names)
    suffix = i < length(const_names) ? "," : ","
    println(io, "    $name$suffix")
  end
  println(io, ")")
  println(io)

  content = String(take!(io))
  mkpath(dirname(output))
  write(output, content)
  println("Generated $(length(groups)) groups with $(sum(length(fs) for (_, fs) in groups)) fields → $output")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
  script_dir = @__DIR__
  org_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(script_dir, "trex.org")
  output_file = length(ARGS) >= 2 ? ARGS[2] : joinpath(script_dir, "src", "trexio_standard_fields.jl")

  if !isfile(org_file)
    error("Cannot find trex.org at: $org_file")
  end

  println("Parsing $org_file ...")
  groups = parse_trex_org(org_file)
  println("Found $(length(groups)) groups:")
  for (g, fs) in groups
    println("  $g: $(length(fs)) fields")
  end
  println()

  generate_fields_file(groups, output_file)
end

main()
