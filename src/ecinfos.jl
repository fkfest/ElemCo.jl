""" Various global infos """
module ECInfos
using Parameters, DocStringExtensions
using ..ElemCo.AbstractEC
using ..ElemCo.Utils
using ..ElemCo.FciDump
using ..ElemCo.MSystem

export ECInfo, setup!, set_options!, parse_orbstring, get_occvirt
export setup_space_fd!, setup_space_ms!, setup_space!
export freeze_core!, freeze_nocc!, freeze_nvirt!, save_space, restore_space!
export n_occ_orbs, n_occb_orbs, n_orbs, n_virt_orbs, n_virtb_orbs, len_spaces
export file_exists, add_file!, copy_file!, delete_file!, delete_files!, delete_temporary_files!
export isalphaspin, space4spin

include("options.jl")

"""
    ECInfo

  Global information for `ElemCo`.

  $(FIELDS)
"""
@with_kw mutable struct ECInfo <: AbstractECInfo
  """ path to scratch directory. """
  scr::String = mktempdir(mkpath(joinpath(tempdir(),"elemcojlscr")))
  """ extension of temporary files. """
  ext::String = ".bin"
  """ output file. """
  out = ""
  """ verbosity level. """
  verbosity::Int = 2
  """ options. """
  options::Options = Options()
  """ molecular system. """
  ms::MSys = MSys()
  """ fcidump. """
  fd::FDump = FDump()
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

  `name` is given by the subspaces involved:
    - `o` for occupied
    - `v` for virtual
    - `O` for occupied-β
    - `V` for virtual-β
    - `m` for (full) MO space
    - `M` for (full) β-MO space
    - `A` for AO basis
    - `a` for active orbitals
    - `c` for closed-shell (doubly-occupied) orbitals
    - `P` for auxiliary orbitals (fitting basis)
    - `L` for auxiliary orbitals (Cholesky decomposition, orthogonal)
    - `X` for auxiliary orbitals (amplitudes decomposition)

  The order of subspaces is important, e.g., `ov` is occupied-virtual, `vo` is virtual-occupied.
  Normally, the first subspaces correspond to subscripts of the tensor. 
  For example, `T_vo` contains the singles amplitudes ``T_{a}^{i}``.
  Disambiguity can be resolved by introducing `^` to separate the subscripts from the superscripts,
  e.g., `d_XX` contains ``\\hat v_{XY}`` and `d_^XX` contains ``\\hat v^{XY}`` integrals.
  """
  files::Dict{String,String} = Dict{String,String}()

  """ ignore various errors. """
  ignore_error::Bool = false
  """ subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':' general. """
  space::Dict{Char,Any} = Dict{Char,Any}()
end

"""
    setup_space_fd!(EC::ECInfo)

  Setup EC.space from fcidump EC.fd.
"""
function setup_space_fd!(EC::ECInfo)
  @assert fd_exists(EC.fd) "EC.fd is not set up!"
  nelec = EC.options.wf.nelec
  charge = EC.options.wf.charge
  ms2 = EC.options.wf.ms2

  norb = headvar(EC.fd, "NORB")
  nelec = (nelec < 0) ? headvar(EC.fd, "NELEC") : nelec
  nelec -= charge
  ms2 = (ms2 < 0) ? headvar(EC.fd, "MS2") : ms2
  orbsym = convert(Vector{Int},headvar(EC.fd, "ORBSYM"))
  setup_space!(EC, norb, nelec, ms2, orbsym)
end

"""
    setup_space_ms!(EC::ECInfo)

  Setup EC.space from molecular system EC.ms.
"""
function setup_space_ms!(EC::ECInfo)
  @assert ms_exists(EC.ms) "EC.ms is not set up!"
  nelec = EC.options.wf.nelec
  charge = EC.options.wf.charge
  ms2 = EC.options.wf.ms2

  norb = guess_norb(EC.ms) 
  nelec = (nelec < 0) ? guess_nelec(EC.ms) : nelec
  nelec -= charge
  ms2 = (ms2 < 0) ? mod(nelec,2) : ms2
  orbsym = ones(Int,norb)
  setup_space!(EC, norb, nelec, ms2, orbsym)
end

"""
    setup_space!(EC::ECInfo, norb, nelec, ms2, orbsym)

  Setup EC.space from `norb`, `nelec`, `ms2`, `orbsym` or `occa`/`occb`.
"""
function setup_space!(EC::ECInfo, norb, nelec, ms2, orbsym)
  occa = EC.options.wf.occa
  occb = EC.options.wf.occb
  SP = EC.space
  println("Number of orbitals: ", norb)
  SP['o'], SP['v'], SP['O'], SP['V'] = get_occvirt(EC, occa, occb, norb, nelec; ms2, orbsym)
  SP[':'] = 1:norb
  return
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
    freeze_core!(EC::ECInfo, core::Symbol, freeze_nocc::Int)

  Freeze `freeze_nocc` occupied orbitals. If `freeze_nocc` is negative: guess the number of core orbitals.

  `core` as in [`MSystem.guess_ncore`](@ref).
"""
function freeze_core!(EC::ECInfo, core::Symbol, freeze_nocc::Int)
  if freeze_nocc < 0
    freeze_nocc = guess_ncore(EC.ms, core)
  end
  freeze_nocc!(EC, freeze_nocc)
  return freeze_nocc
end

"""
    freeze_nocc!(EC::ECInfo, nfreeze::Int)

  Freeze `nfreeze` occupied orbitals.
"""
function freeze_nocc!(EC::ECInfo, nfreeze::Int)
  if nfreeze > n_occ_orbs(EC) || nfreeze > n_occb_orbs(EC) 
    error("Cannot freeze more occupied orbitals than there are.")
  end
  if nfreeze <= 0
    return 0
  end
  println("Freezing ", nfreeze, " occupied orbitals")
  println()
  EC.space['o'] = EC.space['o'][nfreeze+1:end]
  EC.space['O'] = EC.space['O'][nfreeze+1:end]
  return nfreeze
end

"""
    freeze_nvirt!(EC::ECInfo, nfreeze::Int)

  Freeze `nfreeze` virtual orbitals.
"""
function freeze_nvirt!(EC::ECInfo, nfreeze::Int)
  if nfreeze > n_virt_orbs(EC) || nfreeze > n_virtb_orbs(EC) 
    error("Cannot freeze more virtual orbitals than there are.")
  end
  if nfreeze <= 0
    return 0
  end
  println("Freezing ", nfreeze, " virtual orbitals")
  println()
  EC.space['v'] = EC.space['v'][1:end-nfreeze]
  EC.space['V'] = EC.space['V'][1:end-nfreeze]
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
  if maximum(orbsym) > 1 && occursin(".", orbs1)
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
  orblist=Vector{Int}()
  for range in filter(!isempty, split(orbs1,';'))
    if first(range) == ':'
      sym = filter(!isempty, split(range[2:end],'.'))[2]
      range = "1."*sym*range
    end
    firstlast = filter(!isempty, split(range,':'))
    if length(firstlast) == 1
      # add the orbital
      orblist=push!(orblist, symorb2orb(firstlast[1],symoffset))
    else
      length(firstlast) == 2 || error("Someting wrong in range $range in orbstring $orbs")
      firstorb = symorb2orb(firstlast[1],symoffset)
      lastorb = symorb2orb(firstlast[2],symoffset)
      # add the range
      orblist=vcat(orblist,[firstorb:lastorb]...)
    end
  end
  allunique(orblist) || error("Repeated orbitals found in orbstring $orbs")
  return orblist
end

"""
    symorb2orb(symorb::SubString, symoffset::Vector{Int})

  Convert a symorb (like 1.3 [orb.sym]) to an orbital number.
  If no sym given, just return the orbital number converted to Int.
"""
function symorb2orb(symorb::SubString, symoffset::Vector{Int})
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
    get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec; ms2=0, orbsym=Vector{Int})

  Use a +/- string to specify the occupation. If `occbs`=="-", the occupation from `occas` is used (closed-shell).
  If both are "-", the occupation is deduced from `nelec` and `ms2`.
  The optional argument `orbsym` is a vector with length norb of orbital symmetries (1 to 8) for each orbital.
"""
function get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec; ms2=0, orbsym=Vector{Int}())
  @assert(isodd(ms2) == isodd(nelec), "Inconsistency in ms2 (2*S) and number of electrons.")
  if occas != "-"
    occa = parse_orbstring(occas; orbsym)
    if occbs == "-"
      # copy occa to occb
      occb = deepcopy(occa)
    else
      occb = parse_orbstring(occbs; orbsym)
    end
    if length(occa)+length(occb) != nelec && !EC.ignore_error
      error("Inconsistency in OCCA ($occas) and OCCB ($occbs) definitions and the number of electrons ($nelec). Use ignore_error (-f) to ignore.")
    end
  else 
    occa = [1:(nelec+ms2)÷2;]
    occb = [1:(nelec-ms2)÷2;]
  end
  virta = [ i for i in 1:norb if i ∉ occa ]
  virtb = [ i for i in 1:norb if i ∉ occb ]
  if occa == occb
    println("Occupied orbitals:", occa)
  else
    println("Occupied α orbitals:", occa)
    println("Occupied β orbitals:", occb)
  end
  return occa, virta, occb, virtb
end



end #module
