"""
    open_dump(EC::ECInfo)

Open the HDF5 file for the ECInfo object.
Usually the file is open all the time...
"""
function open_dump(EC::ECInfo)
  open_dump(EC.dump)
end

function open_dump(dump::ECDump) 
  dump.file = h5open(dump.filename, "r+")["EC"]
end

"""
    close_dump(EC::ECInfo)

Close the HDF5 file for the ECInfo object.
"""
function close_dump(EC::ECInfo)
  close_dump(EC.dump)
end

function close_dump(dump::ECDump)
  close(HDF5.file(dump.file))
end


"""
    checkndump_system(EC::ECInfo; path::AbstractString="current", fcidump="")

Dump the system to the HDF5 file for the ECInfo object. 
`path` is the path in the HDF5 file where the system is stored.
`fcidump` is the path to the fcidump file.
"""
function checkndump_system(EC::ECInfo; path::AbstractString="current", fcidump="")
  if haskey(EC.dump.file, path)
    file = EC.dump.file[path]
  else
    file = create_group(EC.dump.file, path)
  end
  same_system, same_fcidump = check_system(EC; file, fcidump)
  if !same_system || !same_fcidump
    dump_system(EC; file, fcidump)
  end
  return same_system, same_fcidump
end

"""
    check_system(EC::ECInfo; file::HDF5.Group=EC.dump.file, fcidump="")

Check if the system in the HDF5 file is the same as the one in the ECInfo object.
`file` is the HDF5 file where the system is stored.
`fcidump` is the path to the fcidump file.
"""
function check_system(EC::ECInfo; file::HDF5.Group=EC.dump.file, fcidump="")
  same_system = false
  same_fcidump = false
  if haskey(file, "system")
    sysfile = file["system"]
    sys = fetch_geometry(sysfile)
    same_system = sys ≈ EC.system
    if haskey(sysfile, "fcidump")
      fcidump_old = read(sysfile["fcidump"])
      same_fcidump = (fcidump == "" || isnothing(fcidump)) || (fcidump == fcidump_old)
    else
      same_fcidump = (fcidump == "" || isnothing(fcidump))
    end
  end
  return same_system, same_fcidump
end

"""
    dump_system(EC::ECInfo; path::AbstractString="", file::HDF5.Group=EC.dump.file, fcidump="")

Dump the system to the HDF5 file for the ECInfo object.
`path` is the path in the HDF5 file where the system is stored.
`fcidump` is the path to the fcidump file.
"""
function dump_system(EC::ECInfo; path::AbstractString="", file::HDF5.Group=EC.dump.file, fcidump="")
  h5file = (path == "") ? file : file[path]
  if haskey(h5file, "system")
    delete_object(file, "system")
  end
  sysfile = create_group(h5file, "system")
  dump_geometry(EC; file=sysfile)
  dump_options(EC; file=sysfile)
  if fcidump != "" && !isnothing(fcidump)
    sysfile["fcidump"] = fcidump
  end
end

function dump_geometry(EC::ECInfo; path::AbstractString="", file::HDF5.Group=EC.dump.file)
  h5file = (path == "") ? file : file[path]
  geom = create_group(h5file, "geometry", track_order=true)
  natom = 0
  for at in EC.system
    natom += 1
    dump_atomcentre(geom, at, natom)
  end
end

function dump_atomcentre(file::HDF5.Group, at::ACentre, natom::Int)
  atom = create_group(file, lpad(natom, 3, '0')*":$(atomic_centre_label(at))")
  write(atom, "position", Vector(atomic_position(at)))
  basis = create_group(atom, "basis")
  dump_basis(basis, at.basis)
end

function fetch_geometry(file::HDF5.Group)
  geom = file["geometry"]
  atoms = ACentre[]
  for key in keys(geom)
    at = fetch_atomcentre(geom[key], key)
    push!(atoms, at)
  end
  return MSystem(atoms)
end

function fetch_atomcentre(file::HDF5.Group, key::String)
  iatom, label = parse_atomlabel(key)
  pos = read(file["position"])
  @assert length(pos) == 3 "Invalid position for atom $key"
  basis = fetch_basis(file["basis"])
  return ACentre(label, pos[1], pos[2], pos[3], basis)
end

function parse_atomlabel(atomlabel::String)
    parts = split(atomlabel, ":", limit=2)
    if length(parts) != 2
        error("Invalid key format: $atomlabel. Expected format like '001:He'")
    end
    iatom = parse(Int, parts[1])
    return iatom, parts[2]
end

function fetch_basis(file::HDF5.Group)
  basis = Dict{String,String}()
  for key in keys(file)
    basis[key] = read(file[key])
  end
  return basis
end

function dump_options(EC::ECInfo; path::AbstractString="", file::HDF5.Group=EC.dump.file)
  h5file = (path == "") ? file : file[path]
  # namedtuple from Options
  opts = get_options(EC.options)
  # dump to HDF5
  options2hdf5(h5file, opts)
end

function dump_basis(file::HDF5.Group, basis::Dict)
  dict2hdf5(file, basis)
end

function options2hdf5(file::HDF5.Group, opts::NamedTuple)
  dict2hdf5(file, opts)
end

function dict2hdf5(file::HDF5.Group, dict::Union{Dict,NamedTuple})
  for (key,value) in pairs(dict)
    dict2hdf5(file, string(key), value)
  end
end

function dict2hdf5(file::HDF5.Group, key::String, value::Union{Dict,NamedTuple})
  group = create_group(file, key)
  dict2hdf5(group, value)
end

function dict2hdf5(file::HDF5.Group, key::String, value)
  write(file, key, value)
end

function dict2hdf5(file::HDF5.Group, key::String, value::Symbol)
  write(file, key, string(value))
  attrs(file[key])["type"] = "Symbol"
end