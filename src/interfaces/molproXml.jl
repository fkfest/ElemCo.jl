# XML interface for Molpro mit exportdata/importdata files
using XML

export get_xml_info, get_xml_unique, get_xml_last, get_xml_first
export get_xml_variable, get_xml_variable_values
export get_xml_variable_value_type
export get_xml_molecules, get_xml_geometry
export get_xml_geometry_basis
export set_options_from_xml!
export save_ecvariables_to_file
export MolproInfo
export get_molecule

const MOLPRO_FILE_KEYS = ("XML", "ORBITALS", "ECORBITALS", "ECVARIABLES")

function resolve_exportdata_file(exportdata_file::AbstractString; caller_dir::AbstractString="")
  if isfile(exportdata_file)
    return abspath(exportdata_file)
  elseif !isempty(caller_dir)
    candidate = joinpath(caller_dir, exportdata_file)
    if isfile(candidate)
      return abspath(candidate)
    end
  end
  return exportdata_file
end

function resolve_exportdata_paths!(vardict::Dict{String, String}, exportdata_file::AbstractString)
  base_dir = dirname(abspath(exportdata_file))
  for key in MOLPRO_FILE_KEYS
    if haskey(vardict, key) && !isabspath(vardict[key])
      vardict[key] = normpath(joinpath(base_dir, vardict[key]))
    end
  end
  return vardict
end

"""
    MolproInfo

A structure to hold information parsed from a Molpro export file.
"""
struct MolproInfo
  """ Dictionary of file names and their contents """
  files::Dict{String, String}  
  """ XML document node """
  xml::Node
end

"""
    MolproInfo(exportdata_file::AbstractString)

Constructor for `MolproInfo` that parses an exportdata file from Molpro input.

The exportdata file contains variable definitions in the format:
- `\$VAR=value` - variable definitions
- `!comment` - comments (ignored)
- Other lines terminate parsing

Returns a `MolproInfo` object with parsed variables and XML document.
"""
function MolproInfo(exportdata_file::AbstractString="elemcoil")
  exportdata_file = resolve_exportdata_file(exportdata_file)
  vardict = parse_exportdata_file(exportdata_file)
  resolve_exportdata_paths!(vardict, exportdata_file)
  @assert haskey(vardict, "XML") "No XML variable found in exportdata file"
  xml_doc = read(vardict["XML"], Node) 
  molecule = get_xml_last(xml_doc, "//molecule")
  symmetry = get_xml_unique(molecule, "symmetry")
  irreps = get_xml_info(symmetry, "irreducibleRepresentation")
  if length(irreps) != 1
    error("Expected exactly one irreducible representation, found $(length(irreps)). Use nosym option in Molpro input to disable symmetry!")
  end
  return MolproInfo(vardict, xml_doc)
end

Base.getindex(MI::MolproInfo, key::AbstractString) = MI.files[key]
Base.haskey(MI::MolproInfo, key::AbstractString) = haskey(MI.files, key)

"""
    get_molecule(MI::MolproInfo)

    Get the last molecule from a MolproInfo object.

This function retrieves the last molecule node from the XML document stored in the MolproInfo object.
"""
get_molecule(MI::MolproInfo) = get_xml_last_molecule(MI.xml)

"""
    parse_exportdata_file(exportdata_file::AbstractString)

Parse a Molpro exportdata file and return a dictionary of variables.

The exportdata file contains variable definitions in the format:
- `\$VAR=value` - variable definitions
- `!comment` - comments (ignored)
- Other lines terminate parsing
Returns a dictionary where keys are variable names and values are their corresponding values.
"""
function parse_exportdata_file(exportdata_file::AbstractString)
  vardict = Dict{String, String}()
  open(exportdata_file, "r") do f
    done = false
    while !done
      line = readline(f)
      if Base.eof(f)  # qualified: XML.jl ≥ 0.4 also exports `eof`
        break
      end
      # Strip leading whitespace
      line = lstrip(line)
      # Skip empty and comment lines
      if isempty(line) || line[1] == '!'
        continue
      elseif line[1] == '$'
        # Parse variable definition: $VAR=value
        # Split on space, =, or newline and filter out empty strings
        varval = filter(!isempty, split(line[2:end], r"[ =\n]"))
        if length(varval) != 2
          error("Could not parse the variable pair: $line")
        end
        # Remove quotes from value and store in dictionary
        key = varval[1]
        value = strip(varval[2], [''', '"'])
        vardict[key] = value
      else
        # Something else encountered, finish parsing
        done = true
      end
    end
  end
  return vardict
end

"""
    get_xml_info(node::Node, what::AbstractString)

Get XML information from a node using XPath `what`.

Returns a vector of nodes that match the XPath expression.
If `what` is an empty string, it returns the node itself.
"""
function get_xml_info(node::Node, what::AbstractString)
  return Utils.xpath(what, node)  # qualified: XML.jl ≥ 0.4 also exports `xpath`
end

"""
    get_xml_info(nodes::Vector{Node}, what::AbstractString)

Get XML information from multiple nodes using XPath `what`.

Returns a vector of nodes that match the XPath expression across all nodes.
If `what` is an empty string, it returns the nodes themselves.
"""
function get_xml_info(nodes::Vector{Node}, what::AbstractString)
  results = Node[]
  for node in nodes
    append!(results, Utils.xpath(what, node))  # qualified: XML.jl ≥ 0.4 also exports `xpath`
  end
  return results
end

"""
    get_xml_unique(node::Node, what::AbstractString)

Get a unique XML node from `node` using XPath `what`.

If exactly one node matches, it returns that node.
If no nodes match or multiple nodes match, it raises an error.
"""
function get_xml_unique(node, what::AbstractString)
  results = get_xml_info(node, what)
  if length(results) == 0
    error("No nodes found for unique query '$what'")
  elseif length(results) > 1
    error("Multiple nodes found for unique query '$what'")
  end
  return results[1]
end

"""
    get_xml_last(node::Node, what::AbstractString)

Get the last XML node from `node` using XPath `what`.

If no nodes match, it raises an error.
"""
function get_xml_last(node, what::AbstractString)
  results = get_xml_info(node, what)
  if length(results) == 0
    error("No nodes found for last query '$what'")
  end
  return results[end]
end

"""
    get_xml_first(node::Node, what::AbstractString)

Get the first XML node from `node` using XPath `what`.

If no nodes match, it raises an error.
"""
function get_xml_first(node, what::AbstractString)
  results = get_xml_info(node, what)
  if length(results) == 0
    error("No nodes found for first query '$what'")
  end
  return results[1]
end

"""
    get_xml_variable(node::Node, var::AbstractString)

Get the first variable node with the given name `var` from the XML `node`.

If no such variable exists returns `nothing`. 
If multiple variables with the same name exist, it raises an error.
"""
function get_xml_variable(node::Node, var::AbstractString)
  # Get the first variable node with the given name
  var_node = get_xml_info(node, "//variables/variable[@name='$var']")
  if isempty(var_node)
    return nothing
  elseif length(var_node) > 1
    error("Multiple variable nodes found for '$var'")
  end
  return var_node[1]
end

"""
    get_xml_variable_values(node::Node)

Get the values of a variable node.

Returns a vector of string values for the variable node.
If the node has non-simple children, it raises an error.
"""
function get_xml_variable_values(node::Node)
  values = String[]
  for child in children(node)
    # XML.jl ≥ 0.4 preserves whitespace between elements as Text nodes; skip them
    nodetype(child) == XML.Element || continue
    @assert is_simple(child) "Expected a simple node for variable value"
    push!(values, simple_value(child))
  end
  return values
end

"""
    get_xml_variable_values(node::Node, var::AbstractString)

Get the values of a variable with the given name `var` from the XML `node`.

Returns a tuple of:
- A vector of string values for the variable.
- The type of the variable as a string (e.g., "xsd:double", "xsd:string").

If the variable does not exist, returns an empty vector and "none".
"""
function get_xml_variable_values(node::Node, var::AbstractString)
  var_node = get_xml_variable(node, var)
  if isnothing(var_node)
    return String[], "none"
  end
  @assert haskey(var_node, "type") "Variable node '$var' does not have a 'type' attribute"
  strvals = get_xml_variable_values(var_node)
  return strvals, var_node["type"]
end

"""
    get_xml_variable_values(t::Type{T}, node::Node, var::AbstractString) where T

Get the values of a variable with the given name `var` from the XML `node` as a specific type `T`.

If the variable does not exist, it returns an empty vector of type `T`.
Raises an error if the variable type does not match the expected type.
"""
function get_xml_variable_values(t::Type{T}, node::Node, var::AbstractString) where T
  # Get the type and values
  strvals, vartype = get_xml_variable_values(node, var)
  if vartype == "none"
    return T[]
  end
  if T == Float64
    @assert vartype == "xsd:double" "Expected variable type 'xsd:double' for variable '$var', got '$vartype'"
    return parse.(Float64, strvals)
  elseif T == Int
    @assert vartype == "xsd:double" "Expected variable type 'xsd:double' for variable '$var', got '$vartype'"
    return Int.(parse.(Float64, strvals))
  elseif T == String
    @assert vartype == "xsd:string" "Expected variable type 'xsd:string' for variable '$var', got '$vartype'"
    return strvals
  else
    error("Unsupported type $T for variable '$var'")
  end
end

"""
    get_xml_variable_value(t::Type{T}, node::Node, var::AbstractString) where T

Get the value of a variable with the given name `var` from the XML `node` as a specific type `T`.

If the variable does not exist or has no values, it raises an error.
If the variable has multiple values, it raises an error.
"""
function get_xml_variable_value(t::Type{T}, node::Node, var::AbstractString) where T
  vals = get_xml_variable_values(T, node, var)
  if isempty(vals)
    error("Variable '$var' not found or has no values")
  end
  @assert length(vals) == 1 "Expected exactly one value for variable '$var', got $(length(vals))"
  return vals[1]
end

"""
    get_xml_variable_value_type(node::Node, var::AbstractString)

Get the type and number of values for a variable with the given name `var` from the XML `node`.

  Returns a tuple of:
- The variable type as a string (e.g., "xsd:double", "xsd:string").
- The number of values for the variable.
"""
function get_xml_variable_value_type(node::Node, var::AbstractString)
  strvals, vartype = get_xml_variable_values(node, var)
  return vartype, length(strvals)
end

"""
    get_xml_molecules(node::Node)

Get all molecule nodes from an XML node.

Returns a vector of nodes representing molecules.
"""
function get_xml_molecules(node::Node)
  return get_xml_info(node, "//molecule")
end

"""
    get_xml_last_molecule(node::Node)

Get the last molecule node from an XML node.

Returns the last molecule node found in the XML document.
"""
function get_xml_last_molecule(node::Node)
  molecules = get_xml_molecules(node)
  if isempty(molecules)
    error("No molecule nodes found in the XML document")
  end
  return molecules[end]
end

"""
    get_xml_geometry(molecule::Node)

Get the geometry of a molecule from an XML node.

Returns a tuple of:
- A vector of atom IDs as strings.
- A vector of element types as strings.
- A 3xN matrix of atomic coordinates (in Angstrom), where N is the number of atoms.
"""
function get_xml_geometry(molecule::Node)
  atom_array = get_xml_info(molecule, "//cml:atomArray")
  @assert length(atom_array) == 1 "Expected exactly one atomArray node, found $(length(atom_array)).
  Use a specific molecule node!"
  atoms = get_xml_info(atom_array[1], "cml:atom")
  points = zeros(3, length(atoms))
  ids = String[]
  element_types = String[]
  iatom = 1
  for atom in atoms
    push!(ids, atom["id"])
    push!(element_types, atom["elementType"])
    points[1,iatom] = parse(Float64, atom["x3"])
    points[2,iatom] = parse(Float64, atom["y3"])
    points[3,iatom] = parse(Float64, atom["z3"])
    iatom += 1
  end
  return ids, element_types, points
end

"""
    get_xml_geometry_basis(molecule::Node)

Get the geometry and basis set information for a molecule from an XML node.

Returns a tuple of:
- Geometry as a String.
- Basis set as a String.
"""
function get_xml_geometry_basis(molecule::Node)
  ids, element_types, points = get_xml_geometry(molecule)
  geometry = "$(length(ids))\n\n" 
  for iatom in 1:length(ids)
    geometry *= "$(element_types[iatom]) $(points[1,iatom]) $(points[2,iatom]) $(points[3,iatom])\n"
  end
  basis = get_xml_variable_value(String, molecule, "_BASIS")
  return geometry, basis
end

"""
    set_options_from_xml!(EC::ECInfo, node::Node)

Set the options for an `ECInfo` object from an XML node.

This function extracts the charge, multiplicity, and core electron count from the XML node
and updates the `ECInfo` options accordingly.
"""
function set_options_from_xml!(EC::ECInfo, node::Node)
  charge_vec = get_xml_variable_values(Int, node, "CHARGE")
  if isempty(charge_vec)
    charge = 0
  else
    @assert length(charge_vec) == 1 "Expected exactly one charge value, got $(length(charge_vec))"
    charge = charge_vec[1]
  end
  EC.options.wf.charge = charge
  EC.options.wf.ms2 = get_xml_variable_value(Int, node, "!SPIN")
  core = get_xml_variable_values(Int, node,"_CORE")
  if length(core) == 0
    core = get_xml_variable_values(Int, node,"!DEFCORE")
  end
  if length(core) == 0
    ncore = 0
  else
    ncore = sum(core)
  end
  EC.options.wf.freeze_nocc = ncore
end

"""
    save_ecvariables_to_file(MI::MolproInfo, dict; new=true, prefix="")

Save variables from a dictionary to the `ECVARIABLES` file in the MolproInfo object.

This function writes each key-value pair in the dictionary to the file in the format `key=value`.
If the value is empty, it is skipped. The `prefix` argument allows adding a prefix to each line.
"""
function save_ecvariables_to_file(MI::MolproInfo, dict; new=true, prefix="")
  filename = MI.files["ECVARIABLES"]
  mode = new ? "w" : "a"
  open(filename, mode) do f
    for (key, value) in dict
      if !isempty(value)
        println(f, "$(prefix)$(key)=$(value)")
      end
    end
  end
end
