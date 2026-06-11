"""
           ╭─────────────╮
    Electron Correlation methods
           ╰─────────────╯
"""
module ElemCo

include("version.jl")
include("../lib/TREXIO/src/TREXIO.jl")  # Include standalone TREXIO module
include("../lib/ALPACADecomposition/src/ALPACADecomposition.jl")  # Include standalone ALPACADecomposition module
include("infos/abstractEC.jl")
include("tools/mtensoroperations.jl")
include("tools/descdict.jl")
include("tools/vecdict.jl")
include("tools/outputs.jl")
include("tools/utils.jl")
include("tools/constants.jl")
include("tools/myio.jl")
include("tools/mnpy.jl")
include("tools/qmtensors.jl")
include("integrals/dump.jl")
include("system/elements.jl")
include("system/msystems.jl")
include("system/basisset.jl")
include("system/integrals.jl")

include("infos/ecinfos.jl")

include("interfaces/trexio.jl")
include("system/properties.jl")
include("system/wavefunctions.jl")

include("infos/ecmethods.jl")
include("tools/tensortools.jl")
include("solvers/diis.jl")
include("solvers/davidson.jl")
include("cc/laplace.jl")
include("scf/orbtools.jl")
include("scf/localization.jl")
include("scf/region.jl")
include("scf/fockfactory.jl")
include("integrals/dumptools.jl")
include("integrals/integral_tools.jl")
include("integrals/dfdump.jl")
include("tools/decomptools.jl")
include("fci/fci.jl")
include("cc/cctools.jl")
include("cc/dfcc.jl")
include("cc/cc.jl")
include("cc/dmrg.jl")
include("eom/eom.jl")
include("cc/drivers.jl")

include("scf/bohf.jl")

include("scf/hf.jl")

include("scf/dfmcscf.jl")

include("interfaces/molpro.jl")
include("interfaces/molden.jl")
include("interfaces/vasp.jl")
include("interfaces/interfaces.jl")

try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
using Printf
using Dates
#BLAS.set_num_threads(1)
using PrecompileTools
using Preferences
using .VersionInfo
using .Utils
using .ECInfos
using .QMTensors
using .Properties
using .Wavefunctions
using .ECMethods
using .TensorTools
using .FockFactory
using .CCTools
using .CoupledCluster
using .Drivers
using .DFCoupledCluster
using .FciDumps
using .DumpTools
using .OrbTools
using .OrbLocalization
using .OrbRegion
using .Elements
using .MSystems
using .BasisSets
using .BOHF
using .HF
using .DFMCSCF
using .DfDump
using .DMRG
using .Interfaces
using .TREXIO  # Use the standalone TREXIO module
using .TrexioInterface
using .VaspInterface


export @mainname, @print_input
export @loadfile, @savefile, @copyfile, @deletefile
export @loadwf, @savewf, @copywf, @usewf
export @ECinit, @tryECinit, @setupEC, @set, @opt, @reset, @run, @var2string, @dummy
export @set_default_eltype
# from ECInfos
export ECInfo, ec_eltype, DEFAULT_ELTYPE, set_default_eltype!
export @transform_ints, @write_ints, @dfints, @freeze_orbs, @rotate_orbs, @show_orbs
export @dfhf, @dfhf_positron, @dfuhf, @cc, @dfcc, @dfmp2, @bohf, @bouhf, @dfmcscf
export @localize, @region
export @fci, @ciphi, @sci, @ciϕ
export @import_matrix, @export_molden
export @molpro_input, @molpro_output, @check_molproinfo
# from Utils
export last_energy
# from DescDict
export ODDict
# from Drivers
export extrapolate

"""
    __init__()

  Print the header with the version and the git hash of the current commit.
"""
function __init__()
  draw_line(15)
  println("   ElemCo.jl")
  draw_line(15)
  println("Version: ", version())
  println("Git hash: ", git_hash())
  println("Website: elem.co.il")
  println("Julia version: ",VERSION)
  println("BLAS threads: ",BLAS.get_num_threads())
  println("OpenMP threads: ",Base.Threads.nthreads())
  println("Hostname: ", gethostname())
  println("Scratch directory: ", tempdir())
  println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
  println("""
   ╭──────────────────────────────╮ 
   │        ╭─────────────╮       ├─╮
   │ Electron Correlation methods │ │
   │        ╰─────────────╯       │ │
   ╰─┬────────────────────────────╯ │
     ╰──────────────────────────────╯""")

end

"""
    @mainname(file)

  Return the main name of a file, i.e. the part before the last dot
  and the extension.

  # Examples
```julia
julia> @mainname("~/test.xyz")
("test", "xyz")
```  
"""
macro mainname(file)
  return quote
    mainname($(esc(file)))
  end
end

"""
    @print_input(print_init=false)

  Print the input file content. 

  Can be used to print the input file content to the output.
"""
macro print_input(print_init=false)
  return quote
    if $(esc(print_init))
      __init__()
    end
    try
      print_info(read($(string(__source__.file)), String))
    catch
      print_info("No input file found.")
    end
  end
end

"""
    @loadfile(filename)

  Read file `filename` from `EC.scr` directory.

  # Example
```julia
fock = @loadfile("f_mm")
```
"""
macro loadfile(filename)
  strfilename = clean_exprstring(filename)
  return quote
    strfilename = @var2string($(esc(filename)), $(esc(strfilename)))
    load($(esc(:EC)), strfilename)
  end
end

"""
    @savefile(filename, arr, kwargs...)

  Save array or tuple of arrays `arr` to file `filename` in `EC.scr` directory.

  # Keyword arguments
  - `description::String`: description of the file (default: "tmp").
  - `overwrite::Bool`: overwrite existing file (default: `false`).
"""
macro savefile(filename, arr, kwargs...)
  ekwa = [esc(a) for a in kwargs]
  strfilename = clean_exprstring(filename)
  return quote
    strfilename = @var2string($(esc(filename)), $(esc(strfilename)))
    save!($(esc(:EC)), strfilename, $(esc(arr)); $(ekwa...))
  end
end

"""
    @copyfile(from_file, to_file, kwargs...)

  Copy file `from_file` to `to_file` in `EC.scr` directory.

  # Keyword arguments
  - `overwrite::Bool`: overwrite existing file (default: `false`).
"""
macro copyfile(from_file, to_file, kwargs...)
  ekwa = [esc(a) for a in kwargs]
  strfrom = clean_exprstring(from_file)
  strto = clean_exprstring(to_file)
  return quote
    strfrom = @var2string($(esc(from_file)), $(esc(strfrom)))
    strto = @var2string($(esc(to_file)), $(esc(strto)))
    copy_file!($(esc(:EC)), strfrom, strto; $(ekwa...))
  end
end

"""
    @deletefile(filename)

  Delete file `filename` from `EC.scr` directory.
"""
macro deletefile(filename)
  strfilename = clean_exprstring(filename)
  return quote
    strfilename = @var2string($(esc(filename)), $(esc(strfilename)))
    delete_file!($(esc(:EC)), strfilename)
  end
end

"""
    @loadwf(what...; start=false, state=1)

  Load wavefunction data from the trexio dump file.

  The arguments `what` can be a vector of strings, a string variable or a list of arguments 
  specifying what to load.
  Possible values are: 

  - `all`: load everything available (overrides other options)
  - `orbital_energies`: molecular orbital energies
  - `orbital_occupations`: molecular orbital occupations
  - `amplitudes`: restricted CC amplitudes (T1, T2)
  - `unrestricted_amplitudes`: unrestricted CC amplitudes (T1a, T1b, T2a, T2b, T2ab)
  - `determinants`: selected CI determinants and coefficients

  The loaded data are returned as a dictionary with keys corresponding to the requested data.
  `basis`, `orbitals` and `orbital_type` are always included in the output.

# Keyword Arguments
- `start::Bool=false`: If true, read from `wf.start` file instead of `wf.dump`
- `state::Int=1`: State number for determinants (1 = ground state)
- `OPattern::Type=UInt64`: Orbital pattern type for determinants (use `UInt128` for >64 orbitals)

# Examples
```julia
julia> wf = @loadwf orbital_energies orbital_occupations
julia> wf["basis"]  # basis set information
julia> wf = @loadwf ["orbital_energies", "orbital_occupations"]
julia> wf = @loadwf amplitudes  # load CC amplitudes
julia> wf = @loadwf determinants state=2  # load determinants for excited state
julia> wf = @loadwf all start=true  # load everything from start file
```
"""
macro loadwf(args...)
  what, kwargs = separate_kwargs(args)
  strwhat = String[clean_exprstring(w) for w in what]
  if length(strwhat) == 1
    return quote
      $(esc(:@tryECinit))
      strwhat = @var2string($(esc(what[1])), $(esc(strwhat)), AbstractArray)
      load_wavefunction($(esc(:EC)), strwhat; $(kwargs...))
    end
  else
    return quote
      $(esc(:@tryECinit))
      load_wavefunction($(esc(:EC)), $(esc(strwhat)); $(kwargs...))
    end
  end
end

"""
    @savewf(wf::AbstractDict; state=1)

  Save wavefunction data to the trexio dump file.

  The argument `wf` is a dictionary with the data to be saved.
  Possible keys are:

**Orbital data:**
- `"basis"`: basis set information
- `"orbitals"`: molecular orbitals
- `"rotations"`: orbital rotations (alternative to `"orbitals"`)
- `"orbital_type"`: type of the orbitals (e.g., "RHF", "UHF", "ROHF", "MCSCF")
- `"orbital_energies"`: molecular orbital energies
- `"orbital_occupations"`: molecular orbital occupations

**Restricted CC amplitudes:**
- `"T1"`: singles amplitudes (nvirt × nocc)
- `"T2"`: doubles amplitudes (nvirt × nvirt × nocc × nocc)

**Unrestricted CC amplitudes:**
- `"T1a"`, `"T1b"`: α and β singles amplitudes
- `"T2a"`, `"T2b"`, `"T2ab"`: αα, ββ, and αβ doubles amplitudes

**Selected CI (CIPHI) data:**
- `"determinants"`: vector of determinants
- `"ci_coefficients"`: CI coefficients (vector for single state, matrix for multi-state)

# Keyword Arguments
- `state::Int=1`: State number for determinants (used when `ci_coefficients` is a vector)

# Examples
```julia
julia> wf = @loadwf orbital_energies orbital_occupations
julia> orbs = wf["orbitals"]
[...]
julia> wf1 = Dict("basis"=>wf["basis"], "orbitals"=>orbs, "orbital_type"=>"modified RHF") 
julia> @savewf wf1

julia> # Save amplitudes
julia> @savewf Dict("T1"=>T1, "T2"=>T2)

julia> # Save determinants for excited state
julia> @savewf Dict("determinants"=>dets, "ci_coefficients"=>coeffs) state=2
```
"""
macro savewf(args...)
  positional, kwargs = separate_kwargs(args)
  
  if isempty(positional)
    error("@savewf requires a dictionary argument")
  end
  wf = positional[1]
  
  return quote
    $(esc(:@tryECinit))
    save_wavefunction($(esc(:EC)), $(esc(wf)); $(kwargs...))
  end
end

"""
    @copywf(to_file::AbstractString=""; start=false, state=0)

  Copy wavefunction data from the current trexio dump file to another dump file.

  If `to_file` is not provided, the wavefunction is copied to [`EC.options.wf.store`](@ref ECInfos.WfOptions) file.
  Note: This does not check the contents of the files.

# Keyword Arguments
- `start::Bool=false`: If true, copy from `wf.start` file instead of `wf.dump`.
- `state::Int=0`: State number for determinant files. If 0, copies the main dump file.
                   If >0, copies the state-specific determinant file (e.g., `file_state2.h5`).

# Examples
```julia
julia> @copywf  # copy dump to store
julia> @copywf "backup.h5"  # copy dump to backup file
julia> @copywf start=true  # copy start file to store
julia> @copywf "backup.h5" start=true  # copy start file to backup
julia> @copywf state=2  # copy state 2 determinant file to store
julia> @copywf "state2_backup.h5" state=2  # copy state 2 to specific file
```
"""
macro copywf(args...)
  positional, kwargs = separate_kwargs(args)
  
  if isempty(positional)
    return quote
      $(esc(:@tryECinit))
      copy_wavefunction($(esc(:EC)), ""; $(kwargs...))
    end
  else
    to_file_expr = positional[1]
    to_file = clean_exprstring(to_file_expr)
    return quote
      $(esc(:@tryECinit))
      strto = @var2string($(esc(to_file_expr)), $(esc(to_file)))
      copy_wavefunction($(esc(:EC)), strto; $(kwargs...))
    end
  end
end

"""
    @usewf(from_file::AbstractString=""; start=false, state=0)

  Copy wavefunction data to the current trexio dump file from another dump file, i.e., it does the opposite of [`@copywf`](@ref).

  If `from_file` is not provided, the wavefunction is copied from [`EC.options.wf.store`](@ref ECInfos.WfOptions) file.
  Note: This does not check the contents of the files.

# Keyword Arguments
- `start::Bool=false`: If true, copy to `wf.start` file instead of `wf.dump`.
- `state::Int=0`: State number for determinant files. If 0, copies to the main dump file.
                   If >0, copies to the state-specific determinant file (e.g., `file_state2.h5`).

# Examples
```julia
julia> @usewf  # copy store to dump
julia> @usewf "backup.h5"  # copy from backup file to dump
julia> @usewf start=true  # copy store to start file
julia> @usewf "backup.h5" start=true  # copy backup file to start file
julia> @usewf state=2  # copy store file to state 2 determinant file
julia> @usewf "state2_backup.h5" state=2  # copy specific file to state 2
```
"""
macro usewf(args...)
  positional, kwargs = separate_kwargs(args)
  
  if isempty(positional)
    return quote
      $(esc(:@tryECinit))
      copy_wavefunction($(esc(:EC)), ""; $(kwargs...), reverse=true)
    end
  else
    from_file_expr = positional[1]
    from_file = clean_exprstring(from_file_expr)
    return quote
      $(esc(:@tryECinit))
      strto = @var2string($(esc(from_file_expr)), $(esc(from_file)))
      copy_wavefunction($(esc(:EC)), strto; $(kwargs...), reverse=true)
    end
  end
end

""" 
    @ECinit(T=DEFAULT_ELTYPE[])

  Initialize `EC::ECInfo{T}` and add molecular system and/or fcidump 
  if variables `geometry::String` and `basis::Dict{String,Any}`
  and/or `fcidump::String` are defined.

  `T` is the element type for the fcidump integrals (default: `DEFAULT_ELTYPE[]`, 
  which is `Float64` unless changed via [`set_default_eltype!`](@ref) or [`@set_default_eltype`](@ref)).

  If `EC` is already initialized, it will be overwritten.

  # Examples
```julia
geometry="He 0.0 0.0 0.0"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
@ECinit
# output
Occupied orbitals:[1]

```
```julia
@ECinit ComplexF64  # use complex integrals
```
"""
macro ECinit(T=nothing)
  if isnothing(T)
    ecexpr = :(ECInfo{DEFAULT_ELTYPE[]}())
  else
    ecexpr = :(ECInfo{$(esc(T))}())
  end
  if @istoplevel
    return quote
      const $(esc(:EC)) = $ecexpr
      $(esc(:@setupEC))
    end
  else
    return quote
      $(esc(:EC)) = $ecexpr
      $(esc(:@setupEC))
    end
  end
end

""" 
    @set_default_eltype(T)

  Set the default element type for new `ECInfo` objects.

  # Examples
```julia
@set_default_eltype ComplexF64
@ECinit  # will create ECInfo{ComplexF64}
```
"""
macro set_default_eltype(T)
  return :(set_default_eltype!($(esc(T))))
end

""" 
    @setupEC()

  Setup `EC::ECInfo` with geometry, basis, and fcidump if defined.
"""
macro setupEC()
  return quote
    try
      !isnothing($(esc(:fcidump))) || throw(UndefVarError(:fcidump))
      @assert(typeof($(esc(:fcidump))) <: AbstractString, "fcidump must be a String")
      if fd_origin($(esc(:EC)).fd) != $(esc(:fcidump))
        println("FCIDump: ",$(esc(:fcidump)))
        $(esc(:EC)).fd = read_fcidump($(esc(:fcidump)), ec_eltype($(esc(:EC))))
      end
    catch err
      isa(err, UndefVarError) || rethrow(err)
    end
    try
      (!isnothing($(esc(:geometry))) && !isnothing($(esc(:basis)))) || throw(UndefVarError(:geometry))
      @assert(typeof($(esc(:geometry))) <: AbstractString, "geometry must be a String")
      @assert(typeof($(esc(:basis))) <: Union{AbstractDict, AbstractString}, "basis must be a Dict or a String")
      system = parse_geometry($(esc(:geometry)),$(esc(:basis)))
      if !isapprox(system, $(esc(:EC)).system) && !isempty($(esc(:EC)).fd)
        println("Geometry or basis changed, the integrals will be regenerated.")
        $(esc(:EC)).fd = FDump{ec_eltype($(esc(:EC))),3}()  # reset fcidump
      end
      if !issame(system, $(esc(:EC)).system)
        println("Geometry: ",$(esc(:geometry)))
        println("Basis: ",$(esc(:basis)))
        $(esc(:EC)).system = system
      end
    catch err
      isa(err, UndefVarError) || rethrow(err)
    end
  end
end

""" 
    @tryECinit()

  If `EC::ECInfo` is not yet initialized, run [`@ECinit`](@ref) macro.
"""
macro tryECinit()
  return quote
    runECinit = [false]
    try
      $(esc(:EC)).options
    catch
      runECinit[1] = true
    end
    if runECinit[1]
      $(esc(:@ECinit))
    else
      $(esc(:@setupEC))
    end
  end
end

""" 
    @set(opt, kwargs...)

  Set options for `EC::ECInfo`. 
    
  The first argument `opt` is the name of the option (e.g., `scf`, `cc`, `cholesky`), see [`ECInfos.Options`](@ref).
  The keyword arguments are the options to be set (e.g., `thr=1.e-14`, `maxit=10`).
  The current state of the options can be stored in a variable, e.g., `opt_cc = @set cc`.
  The state can then be restored by `@set cc opt_cc`.
  If `EC` is not already initialized, it will be done. 


  # Examples
```julia
optscf = @set scf thr=1.e-14 maxit=10
@set cc maxit=100
...
@set scf optscf
```
"""
macro set(opt, kwargs...)
  stropt="$opt"
  ekwa = [esc(a) for a in kwargs]
  if length(kwargs) == 1 && (typeof(kwargs[1]) != Expr || kwargs[1].head != :(=)) 
    # if only one argument is provided and it is not a keyword argument
    # then set the option to the value of the argument
    return quote
      $(esc(:@tryECinit))
      if hasproperty($(esc(:EC)).options, Symbol($(esc(stropt))))
        typeof($(ekwa[1])) == typeof($(esc(:EC)).options.$opt) || error("Wrong type of argument in @set")
        $(esc(:EC)).options.$opt = deepcopy($(ekwa[1]))
      else
        error("no such option: ",$(esc(stropt)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if hasproperty($(esc(:EC)).options, Symbol($(esc(stropt))))
        deepcopy(set_options!($(esc(:EC)).options.$opt; $(ekwa...)))
      else
        error("no such option: ",$(esc(stropt)))
      end
    end
  end
end

"""
    @opt(opt, kwargs...)

  Alias for [`@set`](@ref).
"""
var"@opt" = var"@set"

""" 
    @reset(opt)

  Reset options for `opt` to default values.
"""
macro reset(opt)
  stropt="$opt"
  return quote
    $(esc(:@tryECinit))
    if hasproperty($(esc(:EC)).options, Symbol($(esc(stropt))))
      $(esc(:EC)).options.$opt = typeof($(esc(:EC)).options.$opt)()
    else
      error("no such option: ",$(esc(stropt)))
    end
  end
end

""" general runner """
macro run(method, kwargs...)
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@tryECinit))
    $method($(esc(:EC)); $(ekwa...))
  end
end

""" 
    @dfhf(opts_block=nothing)

  Run DF-HF calculation. The orbitals are stored to [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
@dfhf
# with local options:
@dfhf begin
  @set scf maxit=100 thr=1.e-12
  @set wf charge=-1
end
```
"""
macro dfhf(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        if $(esc(:EC)).options.wf.npositron > 0
          dfhf_positron($(esc(:EC)))
        else
          dfhf($(esc(:EC)))
        end
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if $(esc(:EC)).options.wf.npositron > 0
        dfhf_positron($(esc(:EC)))
      else
        dfhf($(esc(:EC)))
      end
    end
  end
end

""" 
    @dfuhf(opts_block=nothing)

  Run DF-UHF calculation. The orbitals are stored to [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
@dfuhf
# with local options:
@dfuhf begin
  @set scf maxit=100
end
```
"""
macro dfuhf(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        dfuhf($(esc(:EC)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      dfuhf($(esc(:EC)))
    end
  end
end

"""
    @dfmcscf(opts_block=nothing)

  Run DF-MCSCF calculation. The orbitals are stored to [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
@dfmcscf
# with local options:
@dfmcscf begin
  @set scf maxit=100
  @set wf active="(4,4)"
end
```
"""
macro dfmcscf(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        dfmcscf($(esc(:EC)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      dfmcscf($(esc(:EC)))
    end
  end
end

"""
    @localize(opts_block=nothing)

  Localize the current orbitals using IBO/Pipek-Mezey/Boys (occupied) and optionally OPAO (virtual).
  
  The orbitals are read from [`WfOptions.start`](@ref ECInfos.WfOptions) and stored
  to [`WfOptions.store`](@ref ECInfos.WfOptions).
  If `start` or `store` is not specified, the orbitals are read from and/or stored back to 
  [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Options (set via `@set loc`)
  - `virtual::Bool`: if `true` (default), also localize virtual orbitals via OPAO.
  - `exponent::Int`: IBO exponent, 2 for Pipek-Mezey, 4 for fourth-moment (default).

  # Examples
```julia
@dfhf
@localize
# with local options:
@localize begin
  @set loc virtual=false exponent=2
end
```
"""
macro localize(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        localize_orbitals($(esc(:EC)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      localize_orbitals($(esc(:EC)))
    end
  end
end

"""
    @region(centers=nothing, opts_block=nothing)

  Build a region-tagged orbital dump from localized occupied orbitals and fragment OPAOs.

`centers` is an optional list of atom indices or center labels. When omitted, the
requested centers are taken from `region.inclusive_centers` and `region.exclusive_centers`.
The macro reads orbitals from
[`WfOptions.start`](@ref ECInfos.WfOptions) when provided, otherwise from
[`WfOptions.dump`](@ref ECInfos.WfOptions), and writes the tagged result to
[`WfOptions.store`](@ref ECInfos.WfOptions) if set, otherwise back to the main dump.

Optionally, a `begin...end` block can be provided to set local `region` or `loc`
options for this call.

# Examples
```julia
@region [1, 2]
@region [:O, :H1] begin
  @set region mode=:exclusive occ_charge_thr=0.25 atom_charge_thr=0.15
end
@region [:C1, :C2, :C3, :C4] begin
  @set region pi=:both pi_occupied=1 pi_virtual=1
end
@region begin
  @set region inclusive_centers=[:H1] exclusive_centers=[:O]
end
```
"""
macro region(args...)
  local_opts_expr = nothing
  centers_expr = :(Any[])

  if isempty(args)
    nothing
  elseif length(args) == 1 && is_options_block(args[1])
    local_opts_expr = parse_options_block(args[1])
  elseif is_options_block(args[end])
    length(args) == 2 || error("@region accepts an optional centers argument and at most one options block")
    centers_expr = args[1]
    local_opts_expr = parse_options_block(args[end])
  elseif length(args) == 1
    centers_expr = args[1]
  else
    error("@region accepts an optional centers argument and at most one options block")
  end

  if isnothing(local_opts_expr)
    return quote
      $(esc(:@tryECinit))
      region_orbitals($(esc(:EC)), $(esc(centers_expr)))
    end
  else
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts_expr) do
        region_orbitals($(esc(:EC)), $(esc(centers_expr)))
      end
    end
  end
end

"""
    @dfints(opts_block=nothing)

  Generate 2 and 4-idx MO integrals using density fitting.
  The MO coefficients are read from [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.
"""
macro dfints(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        dfdump($(esc(:EC)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      dfdump($(esc(:EC)))
    end
  end
end

""" 
    @cc(method, args...)

  Run coupled cluster calculation.

  The type of the method is determined by the first argument (ccsd/ccsd(t)/dcsd etc).
  The method can be specified as a string or as a variable, e.g., 
  `@cc CCSD` or `@cc "CCSD"` or `ccmethod="CCSD";  @cc ccmethod`.
  
  Optionally, a `begin...end` block can be provided as the last argument to set 
  local options for this call. The options are reset after the call completes.
  
  # Keyword arguments
  - `fcidump::String`: fcidump file (default: "", i.e., use integrals from `EC`).
  - `occa::String`: occupied α orbitals (default: "-").
  - `occb::String`: occupied β orbitals (default: "-").

  The occupation strings can be given as a `+` separated list, e.g. `occa = 1+2+3` or equivalently `1-3`. 
  Additionally, the spatial symmetry of the orbitals can be specified with the syntax `orb.sym`, e.g. `occa = "-5.1+-2.2+-4.3"`.

  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
@dfhf
@dfints
@cc ccsd
# with local options:
@cc ccsd begin
  @set wf charge=-1 ms2=1
  @set cc maxit=30
end
```
"""
macro cc(method, args...)
  strmethod = clean_exprstring(method)
  # Check if last argument is an options block
  local_opts_expr = nothing
  if !isempty(args) && is_options_block(args[end])
    local_opts_expr = parse_options_block(args[end])
    kwargs = args[1:end-1]
  else
    kwargs = args
  end
  ekwa = [esc(a) for a in kwargs]
  
  if !isnothing(local_opts_expr)
    # With local options
    if kwarg_provided_in_macro(kwargs, :fcidump)
      return quote
        $(esc(:@tryECinit))
        with_local_options($(esc(:EC)), $local_opts_expr) do
          strmethod = @var2string($(esc(method)), $(esc(strmethod)))
          ccdriver($(esc(:EC)), strmethod; $(ekwa...))
        end
      end
    else
      return quote
        $(esc(:@tryECinit))
        with_local_options($(esc(:EC)), $local_opts_expr) do
          if isempty($(esc(:EC)).fd)
            dfdump($(esc(:EC)))
          end
          strmethod = @var2string($(esc(method)), $(esc(strmethod)))
          ccdriver($(esc(:EC)), strmethod; fcidump="", $(ekwa...))
        end
      end
    end
  else
    # Without local options (original behavior)
    if kwarg_provided_in_macro(kwargs, :fcidump)
      return quote
        $(esc(:@tryECinit))
        strmethod = @var2string($(esc(method)), $(esc(strmethod)))
        ccdriver($(esc(:EC)), strmethod; $(ekwa...))
      end
    else
      return quote
        $(esc(:@tryECinit))
        if isempty($(esc(:EC)).fd)
          $(esc(:@dfints))
        end
        strmethod = @var2string($(esc(method)), $(esc(strmethod)))
        ccdriver($(esc(:EC)), strmethod; fcidump="", $(ekwa...))
      end
    end
  end
end

"""
    @dfcc(method="svd-dcsd", opts_block=nothing)

  Run coupled cluster calculation using density fitted integrals.

  The type of the method is determined by the first argument.
  The method can be specified as a string or as a variable, e.g., 
  `@dfcc SVD-DCSD` or `@dfcc "SVD-DCSD"` or `ccmethod="SVD-DCSD";  @dfcc ccmethod`.
  
  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.
  
  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
@dfhf
@dfcc svd-dcsd
# with local options:
@dfcc svd-dcsd begin
  @set cc maxit=30
end
```
"""
macro dfcc(method="svd-dcsd", opts_block=nothing)
  strmethod = clean_exprstring(method)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        strmethod = @var2string($(esc(method)), $(esc(strmethod)))
        dfccdriver($(esc(:EC)), strmethod)
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      strmethod = @var2string($(esc(method)), $(esc(strmethod)))
      dfccdriver($(esc(:EC)), strmethod)
    end
  end
end

""" 
    @dfmp2(opts_block=nothing)

  Run density-fitted MP2 calculation.

  If `save` is set in [`CcOptions.save`](@ref ECInfos.CcOptions), 
  the MP2 doubles amplitudes are saved to `save`*"_2" file.

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
@dfmp2
# with local options:
@dfmp2 begin
  @set cc save="mp2_amplitudes"
end
```
"""
macro dfmp2(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts) do
        dfccdriver($(esc(:EC)), "MP2")
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      dfccdriver($(esc(:EC)), "MP2")
    end
  end
end

""" 
    @fci(args...)

  Run FCI calculation.

  Optionally, a `begin...end` block can be provided as the last argument to set 
  local options for this call. The options are reset after the call completes.

  # Keyword arguments
  - `occa::String`: occupied α orbitals (default: "-").
  - `occb::String`: occupied β orbitals (default: "-").

  The occupation strings can be given as a `+` separated list, e.g. `occa = 1+2+3` or equivalently `1-3`. 
  Additionally, the spatial symmetry of the orbitals can be specified with the syntax `orb.sym`, e.g. `occa = "-5.1+-2.2+-4.3"`.

  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"6-31g", "jkfit"=>"vdz-jkfit", "mpfit"=>"vdz-mpfit")
@dfhf
@fci
# with local options:
@fci begin
  @set wf charge=-1
end
```
"""
macro fci(args...)
  # Check if last argument is an options block
  local_opts_expr = nothing
  if !isempty(args) && is_options_block(args[end])
    local_opts_expr = parse_options_block(args[end])
    kwargs = args[1:end-1]
  else
    kwargs = args
  end
  ekwa = [esc(a) for a in kwargs]
  
  if !isnothing(local_opts_expr)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts_expr) do
        if isempty($(esc(:EC)).fd)
          dfdump($(esc(:EC)))
        end
        fcidriver($(esc(:EC)); $(ekwa...))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        $(esc(:@dfints))
      end
      fcidriver($(esc(:EC)); $(ekwa...))
    end
  end
end

""" 
    @ciphi(args...)

  Run CIPHI (CIΦ - CI via Perturbative and Heat-Bath Iterative selection) calculation.

  Optionally, a `begin...end` block can be provided as the last argument to set 
  local options for this call. The options are reset after the call completes.

  # Keyword arguments
  - `occa::String`: occupied α orbitals (default: "-").
  - `occb::String`: occupied β orbitals (default: "-").

  The occupation strings can be given as a `+` separated list, e.g. `occa = 1+2+3` or equivalently `1-3`. 
  Additionally, the spatial symmetry of the orbitals can be specified with the syntax `orb.sym`, e.g. `occa = "-5.1+-2.2+-4.3"`.

  `@sci` and `@ciϕ` are aliases for this macro.

  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"6-31g", "jkfit"=>"vdz-jkfit", "mpfit"=>"vdz-mpfit")
@dfhf
@ciphi
# with local options:
@ciphi begin
  @set ciphi epsilon=1.e-4
end
```
"""
macro ciphi(args...)
  # Check if last argument is an options block
  local_opts_expr = nothing
  if !isempty(args) && is_options_block(args[end])
    local_opts_expr = parse_options_block(args[end])
    kwargs = args[1:end-1]
  else
    kwargs = args
  end
  ekwa = [esc(a) for a in kwargs]
  
  if !isnothing(local_opts_expr)
    return quote
      $(esc(:@tryECinit))
      with_local_options($(esc(:EC)), $local_opts_expr) do
        if isempty($(esc(:EC)).fd)
          dfdump($(esc(:EC)))
        end
        fcidriver($(esc(:EC)); $(ekwa...), ciphi=true)
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        $(esc(:@dfints))
      end
      fcidriver($(esc(:EC)); $(ekwa...), ciphi=true)
    end
  end
end

"""
    @sci(args...)

  Alias for [`@ciphi`](@ref).
"""
var"@sci" = var"@ciphi"

"""
    @ciϕ(args...)

  Alias for [`@ciphi`](@ref).
"""
var"@ciϕ" = var"@ciphi"

""" 
    @bohf(opts_block=nothing)

  Run bi-orthogonal HF calculation using FCIDUMP integrals.

  The orbital rotations are stored to [`WfOptions.dump`](@ref ECInfos.WfOptions).
  For open-shell systems (or UHF FCIDUMPs), the BO-UHF energy is calculated.

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
fcidump = "FCIDUMP"
@bohf
# with local options:
@bohf begin
  @set scf maxit=100
end
```
"""
macro bohf(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        error("No FCIDump found.")
      end
      with_local_options($(esc(:EC)), $local_opts) do
        if is_closed_shell($(esc(:EC)))
          bohf($(esc(:EC)))
        else
          bouhf($(esc(:EC)))
        end
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        error("No FCIDump found.")
      end
      if is_closed_shell($(esc(:EC)))
        bohf($(esc(:EC)))
      else
        bouhf($(esc(:EC)))
      end
    end
  end
end

""" 
    @bouhf(opts_block=nothing)

  Run bi-orthogonal UHF calculation using FCIDUMP integrals.

  Optionally, a `begin...end` block can be provided to set local options for this call.
  The options are reset after the call completes.

  # Examples
```julia
fcidump = "FCIDUMP"
@bouhf
# with local options:
@bouhf begin
  @set scf maxit=100
end
```
"""
macro bouhf(opts_block=nothing)
  if !isnothing(opts_block) && is_options_block(opts_block)
    local_opts = parse_options_block(opts_block)
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        error("No FCIDump found.")
      end
      with_local_options($(esc(:EC)), $local_opts) do
        bouhf($(esc(:EC)))
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        error("No FCIDump found.")
      end
      bouhf($(esc(:EC)))
    end
  end
end

"""
    @transform_ints()

  Rotate FCIDump integrals using rotations from [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  as transformation matrices.

  The orbital rotations are read from [`WfOptions.dump`](@ref ECInfos.WfOptions).
  If type of the rotations contains the word `biorthogonal`, 
  the bi-orthogonal orbitals are used.
"""
macro transform_ints()
  return quote
    $(esc(:@tryECinit))
    if isempty($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    CMOl, CMOr = load_left_right_rotations($(esc(:EC)))
    transform_fcidump!($(esc(:EC)).fd, CMOl, CMOr)
  end
end

"""
    @write_ints(file="FCIDUMP", kwargs...)

  Write FCIDump integrals to file `file`.

  # Keyword arguments
  - `tol::Float64`: tolerance for writing integrals (default: `-1.0` - all integrals are written).
  - `format::Symbol`: format for writing integrals (default: `:ascii`). Can be `:npy` for NumPy format.
"""
macro write_ints(file="FCIDUMP", kwargs...)
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@tryECinit))
    if isempty($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    write_fcidump($(esc(:EC)).fd, $file; $(ekwa...))
  end
end

"""
    @dummy(atoms)

  Set atoms as dummy atoms in the system.
  `atoms` is a list of atom indices or atomic symbols.

  After running the macro, only the atoms in the list are set as dummy atoms in the system.

  # Examples
```julia
@dummy [1,2,3]
@dummy ["H1","H2"]
@dummy [1,"H2",:H3]
@dummy [] # unset all dummy atoms
```
"""
macro dummy(atoms)
  return quote
    $(esc(:@tryECinit))
    set_dummy!($(esc(:EC)).system, $(esc(atoms)))
    println("Dummy atoms set to: ", $(esc(atoms)))
    if !isempty($(esc(:EC)).fd)
      println("The integrals will be recalculated.")
      $(esc(:EC)).fd = FDump{ec_eltype($(esc(:EC))),3}() # reset fcidump
    end
  end
end

"""
    @freeze_orbs(freeze_orbs)

  Freeze orbitals in the integrals according to an array or range 
  `freeze_orbs`.

  Alternatively, the orbitals can be specified as a String with the +/- or :/; syntax, e.g.,
  "1-5+7-8", or "1:5;7-8".

  # Examples
```julia
fcidump = "FCIDUMP"
@freeze_orbs 1:5
...
@ECinit
@freeze_orbs [1,2,20,21]
```
"""
macro freeze_orbs(freeze_orbs)
  return quote
    $(esc(:@tryECinit))
    freeze_orbs_in_dump($(esc(:EC)), $(esc(freeze_orbs)))
  end
end

"""
    @rotate_orbs(orb1, orb2, angle, kwargs...)

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  by `angle` (in degrees). For UHF, `spin` can be `:α` or `:β` (keyword argument).
  
  The orbitals are stored to [`WfOptions.store`](@ref ECInfos.WfOptions).

  # Keyword arguments
  - `spin::Symbol`: spin of the orbitals (default: `:α`).

  # Examples
```julia
@dfhf
# swap orbitals 1 and 2
@rotate_orbs 1, 2, 90
```
"""
macro rotate_orbs(orb1, orb2, angle, kwargs...)
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@tryECinit))
    rotate_orbs($(esc(:EC)), $(esc(orb1)), $(esc(orb2)), $(esc(angle)); $(ekwa...))
  end
end

"""
    @show_orbs(range=nothing)

  Show orbitals in the integrals according to an array or range 
  `range`.

  # Examples
```julia
@dfhf
@show_orbs 1:5
```
"""
macro show_orbs(range=nothing)
  return quote
    $(esc(:@tryECinit))
    show_orbitals($(esc(:EC)), $(esc(range)))
  end
end

"""
    @import_matrix(filename)

  Import matrix from file `file`.

  The type of the matrix is determined automatically.
"""
macro import_matrix(filename)
  strfilename = clean_exprstring(filename)
  return quote
    $(esc(:@tryECinit))
    strfilename = @var2string($(esc(filename)), $(esc(strfilename)))
    import_matrix($(esc(:EC)), strfilename)
  end
end

"""
    @export_molden(filename)

  Export current orbitals to Molden file `filename`.
"""
macro export_molden(filename)
  strfilename = clean_exprstring(filename)
  return quote
    $(esc(:@tryECinit))
    strfilename = @var2string($(esc(filename)), $(esc(strfilename)))
    export_molden_orbitals($(esc(:EC)), strfilename)
  end
end

"""
    @molpro_input(filename="elemcoil")

  Initialize the Molpro interface with the given filename.

  It relies on the Molpro XML file to set up the molecule and basis set.
  If the `basis` variable exists, it will be updated with the AO basis set from the XML file.

  See [`MolproInterface`](@ref) for more details on the Molpro interface.
"""
macro molpro_input(filename="elemcoil")
  return quote
    $(esc(:MI)) = MolproInterface.MolproInfo($(esc(filename)))
    mol_node = MolproInterface.get_molecule($(esc(:MI)))
    $(esc(:geometry)), ao_basis = MolproInterface.get_xml_geometry_basis(mol_node)
    newbasis = [true]
    try
      if $(esc(:basis)) isa Dict{String,String}
        $(esc(:basis))["ao"] = ao_basis
        newbasis[1] = false
      end
    catch
    end
    if newbasis[1]
      $(esc(:basis)) = ao_basis
    end
    $(esc(:@ECinit)) # TODO replace with @tryECinit once the mo classes are properly handled
    MolproInterface.set_options_from_xml!($(esc(:EC)), mol_node)
    if haskey($(esc(:MI)), "ORBITALS")
      orbs = MolproInterface.import_orbitals($(esc(:EC)), $(esc(:MI))["ORBITALS"])
      if !isempty(orbs)
        dump_orbitals($(esc(:EC)), SpinMatrix(orbs))
        println("Orbitals imported from Molpro: ", size(orbs, 2), " orbitals.")
      end
    end
  end
end

"""
    @check_molproinfo()

  Check if [`MolproInterface.MolproInfo`](@ref) is initialized and return the files.
  If not initialized, throw an error.
"""
macro check_molproinfo()
  return quote
    try
      $(esc(:MI)).files
    catch
      error("MolproInfo is not initialized. Please run @molpro_input first.")
    end
  end
end

"""
    @molpro_output(ecvariables, kwargs...)

  Save key-value pairs from `ecvariables` to a `ECVARIABLES` file in the [`MolproInterface.MolproInfo`](@ref) object.
  
  The `ecvariables` is a dictionary with the variables to be included in the output.
  The keyword arguments are passed to the [`MolproInterface.save_ecvariables_to_file`](@ref) function.
  Possible keyword arguments include:
  - `prefix::String`: prefix for each variable in the output file (default: "")
  - `new::Bool`: if `true`, create a new file, otherwise append to the existing file (default: `true`)
"""
macro molpro_output(ecvariables, kwargs...)
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@check_molproinfo))
    MolproInterface.save_ecvariables_to_file($(esc(:MI)), $(esc(ecvariables)); $(ekwa...))
  end
end


# Precompilation preferences
# Master toggle: defaults to true for release builds, false for development builds.
# Override via LocalPreferences.toml: [ElemCo] precompile_workload = true/false
const _precompile_workload = @load_preference("precompile_workload", !devel())
# Individual section toggles (only used when master toggle is true):
const _precompile_cc = @load_preference("precompile_cc", true)
const _precompile_fci = @load_preference("precompile_fci", true)
const _precompile_mcscf = @load_preference("precompile_mcscf", false)
const _precompile_complex = @load_preference("precompile_complex", false)

if _precompile_workload
  @setup_workload begin
    savestd = stdout
    redirect_stdout(devnull)
    geometry = "H 0.0 0.0 0.0
                H 0.0 0.0 1.0"
    basis = "vdz"
    @compile_workload begin
      _need_hf = _precompile_cc || _precompile_fci || _precompile_mcscf || _precompile_complex
      if _need_hf
        @dfhf
      end
      if _precompile_cc
        @cc dcsd
        @cc uccsd
        @dfcc svd-dcsd
        @dfmp2
      end
      if _precompile_fci
        @fci
      end
      if _precompile_mcscf
        @set wf ms2=2
        @dfmcscf
      end
      if _precompile_complex
        # Complex precompilation uses FCIDUMP-based workflow
        # (DF-HF doesn't support complex 3-index integrals)
        fd_c = FciDumps.FDump{ComplexF64,3}(EC.fd)
        EC_c = ECInfo{ComplexF64}()
        EC_c.fd = fd_c
        Drivers.ccdriver(EC_c, "dcsd")
        if _precompile_fci
          Drivers.fcidriver(EC_c)
        end
      end
    end
    redirect_stdout(savestd)
  end
end

end #module
