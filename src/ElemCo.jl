"""
           ╭─────────────╮
    Electron Correlation methods
           ╰─────────────╯
"""
module ElemCo

include("abstractEC.jl")
include("utils.jl")
include("constants.jl")
include("myio.jl")
include("mnpy.jl")
include("dump.jl")
include("integrals.jl")
include("msystem.jl")

include("ecinfos.jl")
include("ecmethods.jl")
include("tensortools.jl")
include("diis.jl")
include("orbtools.jl")
include("fockfactory.jl")
include("dumptools.jl")
include("dftools.jl")
include("decomptools.jl")
include("cctools.jl")
include("dfcc.jl")
include("cc.jl")
include("ccdriver.jl")

include("bohf.jl")

include("dfhf.jl")
include("dfdump.jl")

include("dfmcscf.jl")

try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
using Printf
using Dates
#BLAS.set_num_threads(1)
using .Utils
using .ECInfos
using .ECMethods
using .TensorTools
using .FockFactory
using .CCTools
using .CoupledCluster
using .CCDriver
using .DFCoupledCluster
using .FciDump
using .DumpTools
using .OrbTools
using .MSystem
using .BOHF
using .DFHF
using .DFMCSCF
using .DfDump


export @mainname, @print_input
export @loadfile, @savefile, @copyfile
export @ECinit, @tryECinit, @set, @opt, @reset, @run, @method2string
export @transform_ints, @write_ints, @dfints, @freeze_orbs, @rotate_orbs
export @dfhf, @dfuhf, @cc, @dfcc, @bohf, @bouhf

const __VERSION__ = "0.11.1+"

"""
    __init__()

  Print the header with the version and the git hash of the current commit.
"""
function __init__()
  draw_line(15)
  println("   ElemCo.jl")
  draw_line(15)
  println("Version: ", __VERSION__)
  srcpath = @__DIR__
  if isdir(joinpath(srcpath,"..",".git"))
    # get hash from git
    try
      hash = read(`git -C $srcpath rev-parse HEAD`, String)
      println("Git hash: ", hash[1:end-1])
    catch
      # get hash from .git/HEAD
      try
        head = read(joinpath(srcpath,"..",".git","HEAD"), String)
        head = split(head)[2]
        hash = read(joinpath(srcpath,"..",".git",head), String)
        println("Git hash: ", hash[1:end-1])
      catch
        println("Git hash: unknown")
      end
    end
  end
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
    @print_input()

  Print the input file content. 

  Can be used to print the input file content to the output.
"""
macro print_input()
  return quote
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
orbs = @loadfile("C_Am")
```
"""
macro loadfile(filename)
  return quote
    load($(esc(:EC)), $(esc(filename)))
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
  return quote
    save!($(esc(:EC)), $(esc(filename)), $(esc(arr)); $(ekwa...))
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
  return quote
    copy_file!($(esc(:EC)), $(esc(from_file)), $(esc(to_file)); $(ekwa...))
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
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mp2fit"=>"cc-pvdz-rifit")
@ECinit
# output
Occupied orbitals:[1]

```
"""
macro ECinit()
  return quote
    $(esc(:EC)) = ECInfo()
    try
      (!isnothing($(esc(:geometry))) && !isnothing($(esc(:basis)))) || throw(UndefVarError(:geometry))
      println("Geometry: ",$(esc(:geometry)))
      println("Basis: ",$(esc(:basis)))
      $(esc(:EC)).system = parse_geometry($(esc(:geometry)),$(esc(:basis)))
    catch err
      isa(err, UndefVarError) || rethrow(err)
    end
    try
      !isnothing($(esc(:fcidump))) || throw(UndefVarError(:geometry))
      println("FCIDump: ",$(esc(:fcidump)))
      $(esc(:EC)).fd = read_fcidump($(esc(:fcidump)))
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
      $(esc(:EC)).verbosity
    catch
      runECinit[1] = true
    end
    if runECinit[1]
      $(esc(:@ECinit))
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
    @method2string(method, strmethod="")

  Return string representation of `method`.

  If `method` is a String variable, return the value of the variable.
  Otherwise, return the string representation of `method` (or `strmethod` if provided).

  # Examples
```julia
julia> @method2string(CCSD)
"CCSD"
julia> CCSD = "UCCSD";
julia> @method2string(CCSD)
"UCCSD"
```
"""
macro method2string(method, strmethod="")
  if strmethod == ""
    strmethod = replace("$method", " " => "")
  end
  varmethod = :($(esc(method)))
  return quote
    isvar = [false]
    try @assert(typeof($(esc(method))) <: AbstractString)
      isvar[1] = true
    catch
    end
    if isvar[1]
      $varmethod
    else
      $(esc(strmethod))
    end
  end
end

""" 
    @dfhf()

  Run DF-HF calculation. The orbitals are stored to [`WfOptions.orb`](@ref ECInfos.WfOptions).
"""
macro dfhf()
  return quote
    $(esc(:@tryECinit))
    dfhf($(esc(:EC)))
  end
end

""" 
    @dfuhf()

  Run DF-UHF calculation. The orbitals are stored to `WfOptions.orb`.
"""
macro dfuhf()
  return quote
    $(esc(:@tryECinit))
    dfuhf($(esc(:EC)))
  end
end

"""
    @dfints()

  Generate 2 and 4-idx MO integrals using density fitting.
  The MO coefficients are read from [`WfOptions.orb`](@ref ECInfos.WfOptions).
"""
macro dfints()
  return quote
    $(esc(:@tryECinit))
    dfdump($(esc(:EC)))
  end
end

""" 
    @cc(method, kwargs...)

  Run coupled cluster calculation.

  The type of the method is determined by the first argument (ccsd/ccsd(t)/dcsd etc).
  The method can be specified as a string or as a variable, e.g., 
  `@cc CCSD` or `@cc "CCSD"` or `ccmethod="CCSD";  @cc ccmethod`.
  
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
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mp2fit"=>"cc-pvdz-rifit")
@dfhf
@dfints
@cc ccsd
```
"""
macro cc(method, kwargs...)
  strmethod=replace("$method", " " => "")
  ekwa = [esc(a) for a in kwargs]
  if kwarg_provided_in_macro(kwargs, :fcidump)
    return quote
      $(esc(:@tryECinit))
      strmethod = @method2string($(esc(method)), $(esc(strmethod)))
      ccdriver($(esc(:EC)), strmethod; $(ekwa...))
    end
  else
    return quote
      $(esc(:@tryECinit))
      if !fd_exists($(esc(:EC)).fd)
        $(esc(:@dfints))
      end
      strmethod = @method2string($(esc(method)), $(esc(strmethod)))
      ccdriver($(esc(:EC)), strmethod; fcidump="", $(ekwa...))
    end
  end
end

"""
    @dfcc(method="svd-dcsd")

  Run coupled cluster calculation using density fitted integrals.

  The type of the method is determined by the first argument.
  The method can be specified as a string or as a variable, e.g., 
  `@dfcc SVD-DCSD` or `@dfcc "SVD-DCSD"` or `ccmethod="SVD-DCSD";  @dfcc ccmethod`.
  
  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mp2fit"=>"cc-pvdz-rifit")
@dfhf
@dfcc svd-dcsd
```
"""
macro dfcc(method="svd-dcsd")
  strmethod=replace("$method", " " => "")
  return quote
    $(esc(:@tryECinit))
    strmethod = @method2string($(esc(method)), $(esc(strmethod)))
    dfccdriver($(esc(:EC)), strmethod)
  end
end

""" 
    @bohf()

  Run bi-orthogonal HF calculation using FCIDUMP integrals.

  The orbitals are stored to [`WfOptions.orb`](@ref ECInfos.WfOptions).
  For open-shell systems (or UHF FCIDUMPs), the BO-UHF energy is calculated.

  # Examples
```julia
fcidump = "FCIDUMP"
@bohf
```
"""
macro bohf()
  return quote
    $(esc(:@tryECinit))
    if !fd_exists($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    if is_closed_shell($(esc(:EC)))
      bohf($(esc(:EC)))
    else
      bouhf($(esc(:EC)))
    end
  end
end

""" 
    @bouhf()

  Run bi-orthogonal UHF calculation using FCIDUMP integrals.
"""
macro bouhf()
  return quote
    $(esc(:@tryECinit))
    if !fd_exists($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    bouhf($(esc(:EC)))
  end
end

"""
    @transform_ints(type="")

  Rotate FCIDump integrals using [`WfOptions.orb`](@ref ECInfos.WfOptions) as transformation 
  matrices.

  The orbitals are read from [`WfOptions.orb`](@ref ECInfos.WfOptions).
  If type is one of [bo, BO, bi-orthogonal, Bi-orthogonal, biorth, biorthogonal, Biorthogonal], 
  the bi-orthogonal orbitals are used and the left transformation matrix is
  read from [`WfOptions.orb`](@ref ECInfos.WfOptions)*[`WfOptions.left`](@ref ECInfos.WfOptions).
"""
macro transform_ints(type="")
  strtype=replace("$type", " " => "")
  return quote
    $(esc(:@tryECinit))
    if !fd_exists($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    CMOr = load($(esc(:EC)), $(esc(:EC)).options.wf.orb)
    strtype = @method2string($(esc(type)), $(esc(strtype)))
    if strtype ∈ ["bo", "BO", "bi-orthogonal", "Bi-orthogonal", "biorth", "biorthogonal", "Biorthogonal"]
      CMOl = load($(esc(:EC)), $(esc(:EC)).options.wf.orb*$(esc(:EC)).options.wf.left)
    elseif strtype == ""
      CMOl = CMOr
    else
      error("Unknown type in @transform_ints: ", strtype)
    end
    transform_fcidump($(esc(:EC)).fd, CMOl, CMOr)
  end
end

"""
    @write_ints(file="FCIDUMP", tol=-1.0)

  Write FCIDump integrals to file `file`.

If `tol` is negative, all integrals are written, otherwise only integrals with absolute value larger than `tol` are written.
"""
macro write_ints(file="FCIDUMP", tol=-1.0)
  return quote
    $(esc(:@tryECinit))
    if !fd_exists($(esc(:EC)).fd)
      error("No FCIDump found.")
    end
    write_fcidump($(esc(:EC)).fd, $file, $tol)
  end
end

"""
    @freeze_orbs(freeze_orbs)

  Freeze orbitals in the integrals according to an array or range 
  `freeze_orbs`.

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

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.orb`](@ref ECInfos.WfOptions) 
  by `angle` (in degrees). For UHF, `spin` can be `:α` or `:β` (keyword argument).
  
  The orbitals are stored to [`WfOptions.orb`](@ref ECInfos.WfOptions).

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

function run_mcscf()
  geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

  basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")

  @opt wf ms2=2 charge=-2
  E,cMO =  dfmcscf(EC,direct=false)
end

end #module
