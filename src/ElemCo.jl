"""
           ╭─────────────╮
    Electron Correlation methods
           ╰─────────────╯
"""
module ElemCo

include("version.jl")
include("../lib/TREXIO/src/TREXIO.jl")  # Include standalone TREXIO module
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
include("system/wavefunctions.jl")

include("infos/ecmethods.jl")
include("tools/tensortools.jl")
include("solvers/diis.jl")
include("solvers/davidson.jl")
include("cc/laplace.jl")
include("scf/orbtools.jl")
include("scf/fockfactory.jl")
include("integrals/dumptools.jl")
include("integrals/dftools.jl")
include("integrals/decomptools.jl")
include("fci/fci.jl")
include("cc/cctools.jl")
include("cc/dfcc.jl")
include("cc/cc.jl")
include("cc/dmrg.jl")
include("eom/eom.jl")
include("cc/drivers.jl")

include("scf/bohf.jl")

include("scf/dfhf.jl")
include("integrals/dfdump.jl")

include("scf/dfmcscf.jl")

include("interfaces/molpro.jl")
include("interfaces/molden.jl")
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
using .VersionInfo
using .Utils
using .ECInfos
using .QMTensors
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
using .Elements
using .MSystems
using .BasisSets
using .BOHF
using .DFHF
using .DFMCSCF
using .DfDump
using .DMRG
using .Interfaces
using .TREXIO  # Use the standalone TREXIO module
using .TrexioInterface


export @mainname, @print_input
export @loadfile, @savefile, @copyfile
export @loadwf, @savewf, @copywf
export @ECinit, @tryECinit, @setupEC, @set, @opt, @reset, @run, @var2string, @dummy
export @transform_ints, @write_ints, @dfints, @freeze_orbs, @rotate_orbs, @show_orbs
export @dfhf, @dfhf_positron, @dfuhf, @cc, @dfcc, @dfmp2, @bohf, @bouhf, @dfmcscf
export @fci, @hci
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
    @loadwf(what...)

  Load wavefunction data from the trexio dump file.

  The arguments `what` can be a vector of strings, a string variable or a list of arguments 
  specifying what to load.
  Possible values are: 

  - `orbital_energies`: molecular orbital energies
  - `orbital_occupations`: molecular orbital occupations
  - `amplitudes`: coupled cluster amplitudes

  The loaded data are returned as a dictionary with keys corresponding to the requested data.
  `basis`, `orbitals` and `orbital_type` are always included in the output.

  # Examples
```julia
julia> wf = @loadwf orbital_energies orbital_occupations
julia> wf["basis"]  # basis set information
julia> wf = @loadwf ["orbital_energies", "orbital_occupations"]
```
"""
macro loadwf(what...)
  strwhat = String[clean_exprstring(w) for w in what]
  if length(strwhat) == 1
    return quote
      $(esc(:@tryECinit))
      strwhat = @var2string($(esc(what[1])), $(esc(strwhat)), AbstractArray)
      load_wavefunction($(esc(:EC)), strwhat)
    end
  else
    return quote
      $(esc(:@tryECinit))
      load_wavefunction($(esc(:EC)), $(esc(strwhat)))
    end
  end
end

"""
    @savewf(wf::AbstractDict)

  Save wavefunction data to the trexio dump file.

  The argument `wf` is a dictionary with the data to be saved.
  Possible keys are:
  - `basis`: basis set information
  - `orbitals`: molecular orbitals
  - `orbital_type`: type of the orbitals (e.g., "RHF", "UHF", "ROHF", "MCSCF")
  - `orbital_energies`: molecular orbital energies
  - `orbital_occupations`: molecular orbital occupations
  - `amplitudes`: coupled cluster amplitudes

  # Examples
```julia
julia> wf = @loadwf orbital_energies orbital_occupations
julia> orbs = wf["orbitals"]
[...]
julia> wf1 = Dict("basis"=>wf["basis"], "orbitals"=>orbs, "orbital_type"=>"modified RHF") 
julia> @savewf wf1
```
"""
macro savewf(wf)
  return quote
    $(esc(:@tryECinit))
    save_wavefunction($(esc(:EC)), $(esc(wf)))
  end
end

"""
    @copywf(to_file::AbstractString="")

  Copy wavefunction data from the current trexio dump file to another dump file.

  If `to_file` is not provided, the wavefunction is copied to [`EC.options.wf.store`](@ref ECInfos.WfOptions) file.
  Note: This does not check the contents of the files.
"""
macro copywf(to_file="")
  strto = clean_exprstring(to_file)
  return quote
    $(esc(:@tryECinit))
    strto = @var2string($(esc(to_file)), $(esc(strto)))
    copy_wavefunction($(esc(:EC)), strto)
  end
end

""" 
    @ECinit()

  Initialize `EC::ECInfo` and add molecular system and/or fcidump 
  if variables `geometry::String` and `basis::Dict{String,Any}`
  and/or `fcidump::String` are defined.

  If `EC` is already initialized, it will be overwritten.

  # Examples
```julia
geometry="He 0.0 0.0 0.0"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
@ECinit
# output
Occupied orbitals:[1]

```
"""
macro ECinit()
  if @istoplevel
    return quote
      const $(esc(:EC)) = ECInfo()
      $(esc(:@setupEC))
    end
  else
    return quote
      $(esc(:EC)) = ECInfo()
      $(esc(:@setupEC))
    end
  end
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
        $(esc(:EC)).fd = read_fcidump($(esc(:fcidump)))
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
        $(esc(:EC)).fd = TFDump()  # reset fcidump
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
    clean_exprstring(expr)

  Return a clean string from an expression, i.e., without empty spaces and extra parentheses.

  # Examples
```julia
julia> clean_exprstring(:(SVD-CCSD))
"SVD-CCSD"
julia> clean_exprstring(:(eom-svd-df-ccsd(t)))
"eom-svd-df-ccsd(t)"
```
"""
function clean_exprstring(expr)
  if !(expr isa Expr) || expr.head != :call || expr.args[1] ∉ [:-, :+, :*, :/]
    return string(expr)
  end
  return join([clean_exprstring(a) for a in expr.args[2:end]], string(expr.args[1]))
end

"""
    is_options_block(arg)

Check if `arg` is a `begin...end` block for local options.
"""
function is_options_block(arg)
  return arg isa Expr && arg.head == :block
end

"""
    parse_options_block(block::Expr)

Parse a `begin...end` block with local options and return code that constructs 
a NamedTuple for `with_local_options`.

Each line in the block should have one of these formats:
- Set-macro style: `@set option_name key1=val1 key2=val2 ...`
- Function-call style: `option_name(key1=val1, key2=val2, ...)`

Multiple lines for the same option category are merged together.

For example:
```julia
begin
  @set wf charge=-1 ms2=1
  @set cc maxit=30
  @set cc thr=1.e-10
end
# or equivalently:
begin
  wf(charge=-1, ms2=1)
  cc(maxit=30, thr=1.e-10)
end
```
Returns an expression that creates: `(wf=(charge=-1, ms2=1), cc=(maxit=30, thr=1.e-10))`
"""
function parse_options_block(block::Expr)
  @assert block.head == :block "Expected begin...end block"
  # Use a Dict to collect options per category (to allow merging)
  opts_dict = Dict{Symbol, Vector{Expr}}()
  
  for arg in block.args
    arg isa LineNumberNode && continue
    if arg isa Symbol
      # Just option name without any settings - skip or error
      error("Option category '$arg' specified without any settings")
    elseif arg isa Expr
      if arg.head == :macrocall
        macro_sym = arg.args[1]
        if macro_sym == Symbol("@set")
          # Parse @set style: @set wf charge=-1 ms2=1
          # args[1] = @set, args[2] = LineNumberNode, args[3] = option_name, args[4+] = key=value pairs
          opt_name = nothing
          for i in 2:length(arg.args)
            item = arg.args[i]
            item isa LineNumberNode && continue
            if isnothing(opt_name)
              # First non-LineNumberNode argument is the option category
              if item isa Symbol
                opt_name = item
              else
                error("Expected option category name after @set, got: $item")
              end
            elseif item isa Expr && item.head == :(=)
              if !haskey(opts_dict, opt_name)
                opts_dict[opt_name] = Expr[]
              end
              push!(opts_dict[opt_name], Expr(:(=), item.args[1], item.args[2]))
            elseif item isa Expr && item.head == :kw
              if !haskey(opts_dict, opt_name)
                opts_dict[opt_name] = Expr[]
              end
              push!(opts_dict[opt_name], Expr(:(=), item.args[1], item.args[2]))
            else
              error("Expected key=value pair in options block, got: $item")
            end
          end
        else
          error("Unknown macro '$macro_sym' in options block. Use: @set category key=value")
        end
      elseif arg.head == :call
        # Parse function-call style: wf(charge=-1, ms2=1)
        opt_name = arg.args[1]
        for i in 2:length(arg.args)
          kw = arg.args[i]
          if kw isa Expr && kw.head == :(=)
            if !haskey(opts_dict, opt_name)
              opts_dict[opt_name] = Expr[]
            end
            push!(opts_dict[opt_name], Expr(:(=), kw.args[1], kw.args[2]))
          elseif kw isa Expr && kw.head == :kw
            if !haskey(opts_dict, opt_name)
              opts_dict[opt_name] = Expr[]
            end
            push!(opts_dict[opt_name], Expr(:(=), kw.args[1], kw.args[2]))
          else
            error("Expected key=value pair in options block, got: $kw")
          end
        end
      elseif arg.head == :(=)
        # Single assignment like: cc = saved_opts  
        # This would be for restoring, but we don't support it in blocks
        error("Direct assignment not supported in options block. Use: @set category key=value")
      else
        error("Unexpected expression in options block: $arg")
      end
    end
  end
  
  if isempty(opts_dict)
    return :(NamedTuple())
  end
  
  # Build the final NamedTuple expression from the collected options
  opts = Expr[]
  for (opt_name, kwargs) in opts_dict
    inner_tuple = Expr(:tuple, kwargs...)
    push!(opts, Expr(:(=), opt_name, inner_tuple))
  end
  return Expr(:tuple, opts...)
end

"""
    @var2string(var, strvar="", type=AbstractString)

  Return string representation of `var`.

  If `var` is a String (or `type`) variable, return the value of the variable.
  Otherwise, return the string representation of `var` (or `strvar` if provided).

  # Examples
```julia
julia> @var2string(CCSD)
"CCSD"
julia> CCSD = "UCCSD";
julia> @var2string(CCSD)
"UCCSD"
```
"""
macro var2string(var, strvar="", type=AbstractString)
  if strvar == ""
    strvar = clean_exprstring(var)
  end
  valvar = :($(esc(var)))
  return quote
    isvar = [false]
    try @assert(typeof($(esc(var))) <: $(esc(type)))  # check if var is defined and of correct type
      isvar[1] = true
    catch
    end
    if isvar[1]
      $valvar
    else
      $(esc(strvar))
    end
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
    @dfints()

  Generate 2 and 4-idx MO integrals using density fitting.
  The MO coefficients are read from [`WfOptions.dump`](@ref ECInfos.WfOptions).
"""
macro dfints()
  return quote
    $(esc(:@tryECinit))
    dfdump($(esc(:EC)))
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
    @hci(args...)

  Run Heat-bath CI calculation.

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
@hci
# with local options:
@hci begin
  @set fci epsilon1=1.e-4
end
```
"""
macro hci(args...)
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
        fcidriver($(esc(:EC)); $(ekwa...), hci=true)
      end
    end
  else
    return quote
      $(esc(:@tryECinit))
      if isempty($(esc(:EC)).fd)
        $(esc(:@dfints))
      end
      fcidriver($(esc(:EC)); $(ekwa...), hci=true)
    end
  end
end

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


# precompile if not in development mode
if !devel()
  @setup_workload begin
    savestd = stdout
    redirect_stdout(devnull)
    geometry = "H 0.0 0.0 0.0
                H 0.0 0.0 1.0"
    basis = "vdz"
    @compile_workload begin
      @dfhf
      @cc dcsd
      @cc uccsd
      @dfcc svd-dcsd
      @dfmp2
    end
    redirect_stdout(savestd)
  end
end

end #module
