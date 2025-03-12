""" Various global infos """
module ECInfos
using Unitful, UnitfulAtomic
using AtomsBase
using HDF5
using Dates
using DocStringExtensions
using ..ElemCo.VersionInfo
using ..ElemCo.AbstractEC
using ..ElemCo.Utils
using ..ElemCo.FciDumps
using ..ElemCo.MSystem
using ..ElemCo.BasisSets

export ECInfo, setup!, set_options!, parse_orbstring, get_occvirt
export setup_space_fd!, setup_space_system!, setup_space!, reset_wf_info!
export is_closed_shell
export freeze_core!, freeze_nocc!, freeze_nvirt!, save_space, restore_space!
export n_occ_orbs, n_occb_orbs, n_orbs, n_virt_orbs, n_virtb_orbs, len_spaces
export file_exists, add_file!, copy_file!, delete_file!, delete_files!, delete_temporary_files!
export file_description
export isalphaspin, space4spin, spin4space, flipspin
export get_options

include("options.jl")

mutable struct ECDump
  """ file name of the HDF5 dump. """
  filename::String
  """ an HDF5 file with calculation information (for restarts etc). 
  The structure of the HDF5 file is as follows (with `track_order=true`):
```
/EC
  /Molecule1
    <name>
    <geometry>
    /BasisSet1
      <basis set information>
      /State1
        <number of electrons>
        <spin multiplicity>
        <occupation (alpha/beta)>
        <MO coefficients>
        <list of frozen orbitals>
        <CC amplitudes>
        <other information>
      /State2
      ...
    /BasisSet2
    ...
  /Molecule2
    ...
```
  """
  file::HDF5.Group
  function ECDump(filename::AbstractString)
    return new(filename, create_empty_dump(filename))
  end
end

"""
    create_empty_dump(filename::AbstractString)

  Create an empty HDF5 dump file with the given `filename` and information about the package.

  Returns an "EC" group in HDF5 file.
"""
function create_empty_dump(filename)
  file = h5open(filename, "w")
  g = create_group(file, "EC", track_order=true)
  g["version"] = version()
  g["git_hash"] = git_hash()
  g["julia"] = "$VERSION"
  g["hostname"] = gethostname()
  g["scratch"] = tempdir()
  g["date"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
  return file["EC"]
end

"""
    ECInfo

  Global information for `ElemCo`.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct ECInfo <: AbstractECInfo
  """`⟨"system-tmpdir/elemcojlscr/jl_*"⟩` path to scratch directory. """
  scr::String = mktempdir(mkpath(joinpath(tempdir(),"elemcojlscr")))
  """`⟨".bin"⟩` extension of temporary files. """
  ext::String = ".bin"
  """ options. """
  options::Options = Options()
  """ molecular system. """
  system::FlexibleSystem = create_empty_system()
  """ fcidump. """
  fd::TFDump = TFDump()
  """ dump with calculation information (for restarts etc). """
  dump::ECDump = ECDump(joinpath(scr,"ec.h5"))
  """ information about (temporary) files. 
  The naming convention is: `prefix`_ + `name` (+extension `EC.ext` added automatically).
  `prefix` can be:
    - `d` for dressed integrals 
    - `S` for overlap matrix
    - `f` for Fock matrix
    - `e` for orbital energies
    - `D` for density matrix
    - `h` for core Hamiltonian
    - `C` for transformation from one basis to another
    - `T` for amplitudes
    - `U` for trial vectors or Lagrange multipliers

  `name` is given by the subspaces involved:
    - `o` for occupied
    - `v` for virtual
    - `O` for occupied-β
    - `V` for virtual-β
    - `m` for (full) MO space
    - `M` for (full) β-MO space
    - `A` for AO basis
    - `a` for active orbitals
    - `d` for doubly-occupied (closed-shell) orbitals
    - `s` for singly-occupied (open-shell) orbitals
    - `S` for singly-occupied with β electron (open-shell) orbitals
    - `P` for auxiliary orbitals (fitting basis)
    - `L` for auxiliary orbitals (Cholesky decomposition, orthogonal)
    - `X` for auxiliary orbitals (amplitudes decomposition)

  The order of subspaces is important, e.g., `ov` is occupied-virtual, `vo` is virtual-occupied.
  Normally, the first subspaces correspond to subscripts of the tensor. 
  For example, `T_vo` contains the singles amplitudes ``T_{a}^{i}``.
  Disambiguity can be resolved by introducing `^` to separate the subscripts from the superscripts,
  e.g., `d_XX` contains ``\\hat v_{XY}`` and `d_^XX` contains ``\\hat v^{XY}`` integrals.
  Subspaces with multiple characters are possible using `{}`, e.g., `C_vo{bX}` contains ``U_{a}^{i\\bar X}``.
  """
  files::Dict{String,String} = Dict{String,String}()
  """ subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':'/'m'/'M' full MO. """
  space::Dict{Char,Vector{Int}} = Dict{Char,Vector{Int}}()
end

"""
    create_empty_system()

  Create an empty molecular system of type `FlexibleSystem`.
"""
function create_empty_system() 
  fs = isolated_system([:H => [0, 0, 0]u"bohr"])
  deleteat!(fs.particles, 1)
  return fs
end

"""
    reset_wf_info!(EC::ECInfo)

  Reset [`ECInfos.WfOptions`](@ref ECInfos.WfOptions) to default.
"""
function reset_wf_info!(EC::ECInfo)
  EC.options.wf = WfOptions()
end

"""
    setup_space_fd!(EC::ECInfo; verbose=true)

  Setup EC.space from fcidump EC.fd.
"""
function setup_space_fd!(EC::ECInfo; verbose=true)
  @assert fd_exists(EC.fd) "EC.fd is not set up!"
  nelec = EC.options.wf.nelec
  npositron = EC.options.wf.npositron
  charge = EC.options.wf.charge
  ms2 = EC.options.wf.ms2
  @assert npositron == 0 "Positron calculation not supported for post-HF yet."

  norb = headvar(EC.fd, "NORB", Int)
  @assert !isnothing(norb)
  nelec_from_fcidump = headvar(EC.fd, "NELEC", Int)
  @assert !isnothing(nelec_from_fcidump)
  nelec = (nelec < 0) ? nelec_from_fcidump : nelec
  nelec -= charge
  ms2_from_fcidump = headvar(EC.fd, "MS2", Int)
  @assert !isnothing(ms2_from_fcidump)
  ms2_default = (nelec == nelec_from_fcidump) ? ms2_from_fcidump : mod(nelec,2)
  ms2 = (ms2 < 0) ? ms2_default : ms2
  orbsym = headvars(EC.fd, "ORBSYM", Int)
  @assert !isnothing(orbsym)
  setup_space!(EC, norb, nelec, ms2, orbsym; verbose=verbose)
end

"""
    setup_space_system(EC::ECInfo; verbose=true)

  Setup EC.space from molecular system EC.system.
"""
function setup_space_system!(EC::ECInfo; verbose=true)
  @assert system_exists(EC.system) "EC.system is not set up!"
  nelec = EC.options.wf.nelec
  charge = EC.options.wf.charge
  ms2 = EC.options.wf.ms2

  norb = guess_norb(EC) 
  nelec = (nelec < 0) ? guess_nelec(EC.system) : nelec
  nelec -= charge
  ms2 = (ms2 < 0) ? mod(nelec,2) : ms2
  orbsym = ones(Int,norb)
  if verbose
    println("Number of orbitals: ", norb)
    println("Number of electrons: ", nelec)
  end
  if EC.options.wf.npositron > 0
    if verbose
      println("Number of positrons: ", EC.options.wf.npositron)
    end
    @assert ms2 == 0 "Cannot have positrons and spin > 0."
  end
  if verbose
    println("Spin: ", ms2)
  end
  setup_space!(EC, norb, nelec, ms2, orbsym; verbose=verbose)
end

"""
    setup_space!(EC::ECInfo, norb, nelec, ms2, orbsym; verbose=true)

  Setup EC.space from `norb`, `nelec`, `ms2`, `orbsym` or `occa`/`occb`.
"""
function setup_space!(EC::ECInfo, norb, nelec, ms2, orbsym; verbose=true)
  occa = EC.options.wf.occa
  occb = EC.options.wf.occb
  SP = EC.space
  if verbose
    println("Number of orbitals: ", norb)
  end
  SP['o'], SP['v'], SP['O'], SP['V'] = get_occvirt(occa, occb, norb, nelec; ms2, orbsym, EC.options.wf.ignore_error, verbose)
  SP['d'] = intersect(SP['o'], SP['O'])
  SP['s'] = setdiff(SP['o'], SP['d'])
  SP['S'] = setdiff(SP['O'], SP['d'])
  SP[':'] = SP['m'] = SP['M'] = [1:norb;]
  return
end

"""
    is_closed_shell(EC::ECInfo)

  Check if the system is closed-shell 
  according the to the reference occupation and FCIDump.
"""
function is_closed_shell(EC::ECInfo)
  SP = EC.space
  SP_changed = false
  if !haskey(SP, 'o') || !haskey(SP, 'O')
    SP_save = save_space(EC)
    setup_space_fd!(EC)
    SP_changed = true
    SP = EC.space
  end
  cs = (SP['o'] == SP['O'] && !EC.fd.uhf)
  if SP_changed
    restore_space!(EC, SP_save)
  end
  return cs
end

"""
    n_occ_orbs(EC::ECInfo)

  Return number of occupied orbitals (for UHF: α).
"""
function n_occ_orbs(EC::ECInfo)
  return length(EC.space['o'])
end
  
"""
    n_occb_orbs(EC::ECInfo)

  Return number of occupied orbitals (β).
"""
function n_occb_orbs(EC::ECInfo)
  return length(EC.space['O'])
end

"""
    n_orbs(EC::ECInfo)

  Return number of orbitals.
"""
function n_orbs(EC::ECInfo)
  return length(EC.space[':'])
end

"""
    n_virt_orbs(EC::ECInfo)

  Return number of virtual orbitals (for UHF: α).
"""
function n_virt_orbs(EC::ECInfo)
  return length(EC.space['v'])
end

"""
    n_virtb_orbs(EC::ECInfo)

  Return number of virtual orbitals (β).
"""
function n_virtb_orbs(EC::ECInfo)
  return length(EC.space['V'])
end

"""
    len_spaces(EC::ECInfo, spaces::String)

  Return lengths of `spaces` (e.g., "vo" for occupied and virtual orbitals).
"""
function len_spaces(EC::ECInfo, spaces::String)
  return [length(EC.space[sp]) for sp in spaces]
end


"""
    freeze_core!(EC::ECInfo, core::Symbol, freeze_nocc::Int, freeze_orbs=[]; verbose=true)

  Freeze `freeze_nocc` occupied orbitals or orbitals on the `freeze_orbs` list. 
  If `freeze_nocc` is negative and `freeze_orbs` is empty: guess the number of core orbitals.

  `core` as in [`MSystem.guess_ncore`](@ref).
"""
function freeze_core!(EC::ECInfo, core::Symbol, freeze_nocc::Int, freeze_orbs=[]; verbose=true)
  if freeze_nocc < 0 && isempty(freeze_orbs)
    freeze_orbs = 1:guess_ncore(EC.system, core)
  elseif freeze_nocc >= 0 && isempty(freeze_orbs)
    freeze_orbs = 1:freeze_nocc
  elseif freeze_nocc >= 0 && !isempty(freeze_orbs)
    error("Cannot specify both freeze_nocc and freeze_orbs in freeze_core!.")
  end
  freeze_nocc!(EC, freeze_orbs; verbose=verbose)
  return length(freeze_orbs)
end

"""
    freeze_nocc!(EC::ECInfo, freeze; verbose=true)

  Freeze occupied orbitals from the `freeze` list.
"""
function freeze_nocc!(EC::ECInfo, freeze; verbose=true)
  nfreeze = length(freeze)
  if nfreeze != length(intersect(EC.space['o'],freeze)) || nfreeze != length(intersect(EC.space['O'],freeze)) 
    error("Cannot freeze more occupied orbitals than there are.")
  end
  if isempty(freeze)
    return 0
  end
  if verbose
    println("Freezing ", nfreeze, " occupied orbitals")
    println()
  end
  setdiff!(EC.space['o'], freeze)
  setdiff!(EC.space['O'], freeze)
  return nfreeze
end

"""
    freeze_nvirt!(EC::ECInfo, nfreeze::Int, freeze_orbs=[]; verbose=true)

  Freeze `nfreeze` virtual orbitals or orbitals on the `freeze_orbs` list.
"""
function freeze_nvirt!(EC::ECInfo, nfreeze::Int, freeze_orbs=[]; verbose=true)
  if nfreeze > 0 
    if isempty(freeze_orbs)
      freeze_orbs = 1:nfreeze
    else
      error("Cannot specify both nfreeze and freeze_orbs in freeze_nvirt!.")
    end
  end
  nfreeze = length(freeze_orbs)
  if nfreeze != length(intersect(EC.space['v'],freeze_orbs)) || nfreeze != length(intersect(EC.space['V'],freeze_orbs)) 
    error("Cannot freeze more virtual orbitals than there are.")
  end
  if isempty(freeze_orbs)
    return 0
  end
  if verbose
    println("Freezing ", nfreeze, " virtual orbitals")
    println()
  end
  setdiff!(EC.space['v'], freeze_orbs)
  setdiff!(EC.space['V'], freeze_orbs)
  return nfreeze
end

"""
    save_space(EC::ECInfo)

  Save the current subspaces of space.
"""
function save_space(EC::ECInfo)
  return deepcopy(EC.space)
end

"""
    restore_space!(EC::ECInfo, space)

  Restore the space.
"""
function restore_space!(EC::ECInfo, space)
  EC.space = deepcopy(space)
end

""" 
    set_options!(opt; kwargs...)

  Set options for option `opt` using keyword arguments.
"""
function set_options!(opt; kwargs...)
  for (key,value) in kwargs
    if hasproperty(opt, key)
      setproperty!(opt, key, value)
    else
      error("invalid option name: $key")
    end
  end
  return opt
end

""" 
    set_options!(opt, allopts)

  Set options for option `opt` from `allopts`.
"""
function set_options!(opt, allopts)
  if typeof(allopts) == typeof(opt)
    opt .= allopts 
  else
    error("Argument has to be of type $(typeof(opt))")
  end
  return opt
end

"""
    file_exists(EC::ECInfo, name::String)

  Check if file `name` exists in ECInfo.
"""
function file_exists(EC::ECInfo, name::String)
  return haskey(EC.files, name)
end

"""
    add_file!(EC::ECInfo, name::String, descr::String; overwrite=false)

  Add file `name` to ECInfo with (space-separated) descriptions `descr`.
  Possible description: `tmp` (temporary).
"""
function add_file!(EC::ECInfo, name::String, descr::String; overwrite=false)
  if !file_exists(EC, name) || overwrite
    EC.files[name] = descr
  else
    error("File $name already exists in ECInfo. Use overwrite=true to overwrite.")
  end
end

"""
    file_description(EC::ECInfo, name::String)

  Return description of file `name` in ECInfo.
"""
function file_description(EC::ECInfo, name::String)
  if !file_exists(EC, name)
    error("File $name is not registered in ECInfo.")
  end
  return EC.files[name]
end

"""
    copy_file!(EC::ECInfo, from::AbstractString, to::AbstractString; overwrite=false)

  Copy file `from` to `to`.
"""
function copy_file!(EC::ECInfo, from::AbstractString, to::AbstractString; overwrite=false)
  if !file_exists(EC, from)
    error("File $from is not registered in ECInfo.")
  end
  if !file_exists(EC, to) || overwrite
    EC.files[to] = EC.files[from]
    cp(joinpath(EC.scr, from*EC.ext), joinpath(EC.scr, to*EC.ext), force=true)
  else
    error("File $to already exists in ECInfo. Use overwrite=true to overwrite.")
  end
end

"""
    delete_file!(EC::ECInfo, name::AbstractString)

  Delete file `name` from ECInfo.
"""
function delete_file!(EC::ECInfo, name::AbstractString)
  if !file_exists(EC, name)
    error("File $name is not registered in ECInfo.")
  end
  rm(joinpath(EC.scr, name*EC.ext), force=true)
  delete!(EC.files, name)
end

"""
    delete_files!(EC::ECInfo, which::AbstractString)

  Delete files in ECInfo which match description in `which`.

  `which` can be a space-separated string of descriptions (then all descriptions have to match)
  Examples: 
  - `delete_files!(EC, "tmp")` deletes all temporary files.
  - `delete_files!(EC, "tmp orbs")` deletes all temporary files with additional description "orbs"
"""
function delete_files!(EC::ECInfo, which::AbstractString)
  for (name,descr) in EC.files
    delete = all([w in split(descr) for w in split(which)])
    if delete
      rm(joinpath(EC.scr, name*EC.ext), force=true)
      delete!(EC.files, name)
    end
  end
end

"""
    delete_files!(EC::ECInfo, which::AbstractArray{String})

  Delete files in ECInfo which match any description in array `which`.

  Examples: 
  - `delete_files!(EC, ["tmp","orbs"])` deletes all temporary files and all files with description "orbs".
  - `delete_files!(EC, ["tmp orbs","tmp2"])` deletes all temporary files with description "orbs" and all files with description "tmp2".
"""
function delete_files!(EC::ECInfo, which::AbstractArray{String})
  for w in which
    delete_files!(EC, w)
  end
end

"""
    delete_temporary_files!(EC::ECInfo)

  Delete all temporary files in ECInfo.  
"""
function delete_temporary_files!(EC::ECInfo)
  delete_files!(EC, "tmp")
end

"""
    isalphaspin(sp1::Char,sp2::Char)

  Try to guess spin of an electron: lowcase α, uppercase β, non-letters skipped.
  Return true for α spin.  Throws an error if cannot decide.
"""
function isalphaspin(sp1::Char, sp2::Char)
  if isletter(sp1)
    return islowercase(sp1)
  elseif isletter(sp2)
    return islowercase(sp2)
  else
    error("Cannot guess spincase for $sp1 $sp2 . Specify the spincase explicitly!")
  end
end

"""
    space4spin(sp::Char, alpha::Bool)
    
  Return the space character for a given spin.
  `sp` on input has to be lowercase.
"""
function space4spin(sp::Char, alpha::Bool)
  if alpha
    return sp
  else
    return uppercase(sp)
  end
end

"""
    spin4space(sp::Char)

  Return spin for a given space character.
"""
function spin4space(sp::Char)
  return islowercase(sp) ? :α : :β
end

"""
    flipspin(sp::Char)

  Flip spin for a given space character.
"""
function flipspin(sp::Char)
  return islowercase(sp) ? uppercase(sp) : lowercase(sp)
end

"""
    parse_orbstring(orbs::String; orbsym=Vector{Int})

  Parse a string specifying some list of orbitals, e.g., 
  `-3+5-8+10-12` → `[1 2 3 5 6 7 8 10 11 12]`
  or use ':' and ';' instead of '-' and '+', respectively.
"""
function parse_orbstring(orbs::String; orbsym=Vector{Int}())
  # make it in julia syntax
  orbs1 = replace(orbs,"-"=>":")
  orbs1 = replace(orbs1,"+"=>";")
  orbs1 = replace(orbs1," "=>"")
  if length(orbsym) == 0
    symoffset = zeros(Int,1)
  elseif maximum(orbsym) > 1 && occursin(".", orbs1)
    @assert(issorted(orbsym),"Orbital symmetries are not sorted. Specify occa and occb without symmetry.")
    symoffset = zeros(Int, maximum(orbsym))
    symlist = zeros(Int, maximum(orbsym))
    for sym in eachindex(symlist)
      symlist[sym] = count(isequal(sym), orbsym)
    end
    for iter in eachindex(symoffset)
      if iter == 1
        continue
      end
      symoffset[iter] = sum(symlist[1:iter-1])
    end
    symlist = nothing
  elseif prod(orbsym) == 1 && occursin(".",orbs1)
    error("FCIDUMP without sym but orbital occupations with sym.")
  else
    symoffset = zeros(Int,1)
  end
  occursin(r"^[0-9:;.]+$",orbs1) || error("Use only `0123456789:;+-.` characters in the orbstring: $orbs")
  if first(orbs1) == ':'
    orbs1 = "1"*orbs1
  end
  orblist=Int[]
  for range in filter(!isempty, split(orbs1,';'))
    if first(range) == ':'
      sym = filter(!isempty, split(range[2:end],'.'))[2]
      range = "1."*sym*range
    end
    firstlast = filter(!isempty, split(range,':'))
    if length(firstlast) == 1
      # add the orbital
      push!(orblist, symorb2orb(firstlast[1],symoffset))
    else
      length(firstlast) == 2 || error("Someting wrong in range $range in orbstring $orbs")
      firstorb = symorb2orb(firstlast[1],symoffset)
      lastorb = symorb2orb(firstlast[2],symoffset)
      # add the range
      append!(orblist, [firstorb:lastorb;])
    end
  end
  allunique(orblist) || error("Repeated orbitals found in orbstring $orbs")
  sort!(orblist)
  return orblist
end

"""
    symorb2orb(symorb::AbstractString, symoffset::Vector{Int})

  Convert a symorb (like 1.3 [orb.sym]) to an orbital number.
  If no sym given, just return the orbital number converted to Int.
"""
function symorb2orb(symorb::AbstractString, symoffset::Vector{Int})
  if occursin(".",symorb)
    orb, sym = filter(!isempty, split(symorb,'.'))
    @assert(parse(Int,sym) <= length(symoffset),"Symmetry label $sym larger than maximum of orbsym vector.")
    orb = parse(Int,orb)
    orb += symoffset[parse(Int,sym)]
    return orb
  else
    return parse(Int,symorb)
  end
end

"""
    get_occvirt(occas::String, occbs::String, norb, nelec; ms2=0, orbsym=Vector{Int}, ignore_error=false, verbose=true)

  Use a +/- string to specify the occupation. If `occbs`=="-", the occupation from `occas` is used (closed-shell).
  If both are "-", the occupation is deduced from `nelec` and `ms2`.
  The optional argument `orbsym` is a vector with length norb of orbital symmetries (1 to 8) for each orbital.
"""
function get_occvirt(occas::String, occbs::String, norb::Int, nelec::Int; 
                     ms2=0, orbsym=Vector{Int}(), ignore_error=false, verbose=true)
  @assert(isodd(ms2) == isodd(nelec), "Inconsistency in ms2 (2*S) and number of electrons.")
  occa = Int[]
  occb = Int[]
  if occas != "-"
    append!(occa, parse_orbstring(occas; orbsym))
    if occbs == "-"
      # copy occa to occb
      append!(occb, occa)
    else
      append!(occb, parse_orbstring(occbs; orbsym))
    end
    if length(occa)+length(occb) != nelec && !ignore_error
      error("Inconsistency in OCCA ($occas) and OCCB ($occbs) definitions and the number of electrons ($nelec). Use ignore_error wf option to ignore.")
    end
  else 
    append!(occa, [1:(nelec+ms2)÷2;])
    append!(occb, [1:(nelec-ms2)÷2;])
  end
  virta = [ i for i in 1:norb if i ∉ occa ]
  virtb = [ i for i in 1:norb if i ∉ occb ]
  if verbose
    if occa == occb
      println("Occupied orbitals:", occa)
    else
      println("Occupied α orbitals:", occa)
      println("Occupied β orbitals:", occb)
    end
  end
  return occa, virta, occb, virtb
end


include("ecdump.jl")

end #module
