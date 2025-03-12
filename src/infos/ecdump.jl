
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

function dump_system(EC::ECInfo)
  # check last system (i.e., `system#`) in dump.file:
  # if it exists, create `system# + 1`
  # if not, create `system001`
  keys = keys(EC.dump.file)
  # find all groups starting with "system"
  system_keys = sort(filter(x -> startswith(x, "system"), keys))
  last_system_number = 0
  if length(system_keys) > 0
    # get the number of the last system
    last_system_number = parse(Int, system_keys[end][7:end])
  end
  # create group for system
  group = create_group(EC.dump.file, "system" * lpad(last_system_number + 1, 3, '0'))
  EC.dump.file = group
end

function dump_geometry(EC::ECInfo)
  geom = create_group(EC.dump.file, "geometry")
  natom = 0
  for at in EC.system
    natom += 1
    dump_atomcentre(geom, at, natom)
  end
end

function dump_atomcentre(file::HDF5.Group, at::Atom, natom::Int)
  atom = create_group(file, lpad(natom, 3, '0')*":$(atomic_centre_symbol(at))")
  write(atom, "position", Vector(atomic_position(at)))
  basis = create_group(atom, "basis")
  basis2hdf5(basis, at[:basis])
end

function dump_options(EC::ECInfo)
  # namedtuple from Options
  opts = get_options(EC.options)
  # dump to HDF5
  options2hdf5(EC.dump.file, opts)
end

function basis2hdf5(file::HDF5.Group, basis::Dict)
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