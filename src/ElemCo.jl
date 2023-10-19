"""
       ╭─────────────╮
Electron Correlation methods
       ╰─────────────╯
"""
module ElemCo

include("abstractEC.jl")
include("utils.jl")
include("myio.jl")
include("mnpy.jl")
include("dump.jl")
include("integrals.jl")
include("msystem.jl")

include("ecinfos.jl")
include("ecmethods.jl")
include("tensortools.jl")
include("fockfactory.jl")
include("diis.jl")
include("orbtools.jl")
include("dftools.jl")
include("decomptools.jl")
include("cctools.jl")
include("dfcc.jl")
include("cc.jl")

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
#BLAS.set_num_threads(1)
using .Utils
using .ECInfos
using .ECMethods
using .TensorTools
using .FockFactory
using .CoupledCluster
using .DFCoupledCluster
using .FciDump
using .MSystem
using .BOHF
using .DFHF
using .DFMCSCF
using .DfDump


export ECdriver 
export @mainname
export @loadfile, @savefile, @copyfile
export @ECinit, @tryECinit, @opt, @run, @dfhf, @dfints, @cc, @svdcc

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
      println("Geometry: ",$(esc(:geometry)))
      println("Basis: ",$(esc(:basis)))
      $(esc(:EC)).ms = MSys($(esc(:geometry)),$(esc(:basis)))
    catch err
      isa(err, UndefVarError) || rethrow(err)
    end
    try
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
      $(esc(:EC)).ignore_error
    catch
      runECinit[1] = true
    end
    if runECinit[1]
      $(esc(:@ECinit))
    end
  end
end

""" 
    @opt(what, kwargs...)

  Set options for `EC::ECInfo`. 
    
  The first argument `what` is the name of the option (e.g., `scf`, `cc`, `cholesky`).
  The keyword arguments are the options to be set (e.g., `thr=1.e-14`, `maxit=10`).
  The current state of the options can be stored in a variable, e.g., `opt_cc = @opt cc`. 
  If `EC` is not already initialized, it will be done. 


  # Examples
```julia
@opt scf thr=1.e-14 maxit=10
@opt cc maxit=100
```
"""
macro opt(what, kwargs...)
  strwhat="$what"
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@tryECinit))
    if hasproperty($(esc(:EC)).options, Symbol($(esc(strwhat))))
      set_options!($(esc(:EC)).options.$what; $(ekwa...))
    else
      error("no such option: ",$(esc(strwhat)))
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
    @dfhf()

  Run DFHF calculation. The orbitals are stored to `WfOptions.orb`.
"""
macro dfhf()
  return quote
    $(esc(:@tryECinit))
    dfhf($(esc(:EC)))
  end
end

"""
    @dfints()

  Generate 2 and 4-idx MO integrals using density fitting.
  The MO coefficients are read from `WfOptions.orbs`.
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

  The type of the method is determined by the first argument (ccsd/ccsd(t)/dcsd etc)
  
  # Keyword arguments
  - `fcidump::String`: fcidump file (default: "", i.e., use integrals from `EC`).
  - `occa::String`: occupied α orbitals (default: "-").
  - `occb::String`: occupied β orbitals (default: "-").

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
      ECdriver($(esc(:EC)), $(esc(strmethod)); $(ekwa...))
    end
  else
    return quote
      $(esc(:@tryECinit))
      if !fd_exists($(esc(:EC)).fd)
        $(esc(:@dfints))
      end
      ECdriver($(esc(:EC)), $(esc(strmethod)); fcidump="", $(ekwa...))
    end
  end
end

"""
    @svdcc(method="dcsd")

  Run coupled cluster calculation with SVD decomposition of the amplitudes.

  The type of the method is determined by the first argument (dcsd/dcd)
  
  # Examples
```julia
geometry="bohr
O      0.000000000    0.000000000   -0.130186067
H1     0.000000000    1.489124508    1.033245507
H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mp2fit"=>"cc-pvdz-rifit")
@dfhf
@svdcc
```
"""
macro svdcc(method="dcsd")
  strmethod=replace("$method", " " => "")
  return quote
    calc_svd_dc($(esc(:EC)), $(esc(strmethod)))
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

"""
    is_closed_shell(EC::ECInfo)

  Check if the system is closed-shell 
  according the to the reference occupation and FCIDump.
"""
function is_closed_shell(EC::ECInfo)
  SP = EC.space
  closed_shell = (SP['o'] == SP['O'] && !EC.fd.uhf)
  addname=""
  if !closed_shell
    addname = "U"
  end
  return closed_shell, addname
end

""" 
    calc_fock_matrix(EC::ECInfo, closed_shell)

  Calculate fock matrix from FCIDump
"""
function calc_fock_matrix(EC::ECInfo, closed_shell)
  t1 = time_ns()
  if closed_shell
    fock = gen_fock(EC)
    save!(EC, "f_mm", fock)
    save!(EC, "f_MM", fock)
    eps = diag(fock)
    println("Occupied orbital energies: ", eps[EC.space['o']])
    save!(EC, "e_m", eps)
    save!(EC, "e_M", eps)
  else
    fock = gen_fock(EC, :α)
    eps = diag(fock)
    println("Occupied α orbital energies: ", eps[EC.space['o']])
    save!(EC, "f_mm", fock)
    save!(EC, "e_m", eps)
    fock = gen_fock(EC, :β)
    eps = diag(fock)
    println("Occupied β orbital energies: ", eps[EC.space['O']])
    save!(EC,"f_MM", fock)
    save!(EC,"e_M", eps)
  end
  t1 = print_time(EC,t1,"fock matrix",1)
end

""" 
    calc_HF_energy(EC::ECInfo, closed_shell)

  Calculate HF energy from FCIDump and EC info. 
"""
function calc_HF_energy(EC::ECInfo, closed_shell)
  SP = EC.space
  if closed_shell
    ϵo = load(EC,"e_m")[SP['o']]
    EHF = sum(ϵo) + sum(diag(integ1(EC.fd))[SP['o']]) + EC.fd.int0
  else
    ϵo = load(EC,"e_m")[SP['o']]
    ϵob = load(EC,"e_M")[SP['O']]
    EHF = 0.5*(sum(ϵo)+sum(ϵob) + sum(diag(integ1(EC.fd, :α))[SP['o']]) + sum(diag(integ1(EC.fd, :β))[SP['O']])) + EC.fd.int0
  end
  return EHF
end

""" 
    ECdriver(EC::ECInfo, methods; fcidump="FCIDUMP", occa="-", occb="-")

  Run electronic structure calculation for `EC::ECInfo` using methods `methods::String`.

  The integrals are read from `fcidump::String` (default: "FCIDUMP").
  If `fcidump::String` is empty, the integrals from `EC.fd` are used.
  The occupied α orbitals are given by `occa::String` (default: "-").
  The occupied β orbitals are given by `occb::String` (default: "-").
  If `occb::String` is empty, the occupied β orbitals are the same as the occupied α orbitals (closed-shell case).
"""
function ECdriver(EC::ECInfo, methods; fcidump="FCIDUMP", occa="-", occb="-")
  t1 = time_ns()
  method_names = split(methods)
  if occa != "-"
    EC.options.wf.occa = occa
  end
  if occb != "-"
    EC.options.wf.occb = occb
  end
  if fcidump != ""
    # read fcidump intergrals
    EC.fd = read_fcidump(fcidump)
    t1 = print_time(EC,t1,"read fcidump",1)
  end
  setup_space_fd!(EC)

  closed_shell, addname = is_closed_shell(EC)

  calc_fock_matrix(EC, closed_shell)
  EHF = calc_HF_energy(EC, closed_shell)
  println(addname*"HF energy: ",EHF)
  flush(stdout)

  SP = EC.space
  for mname in method_names
    println()
    println("Next method: ",mname)
    ecmethod = ECMethod(mname)
    if ecmethod.unrestricted
      add2name = "U"
      closed_shell_method = false
    else
      add2name = addname
      closed_shell_method = closed_shell
      ecmethod.unrestricted = !closed_shell
    end
    # at the moment we always calculate MP2 first
    # calculate MP2
    if closed_shell_method
      EMp2 = calc_MP2(EC)
    else
      EMp2 = calc_UMP2(EC)
    end
    println(add2name*"MP2 correlation energy: ",EMp2)
    println(add2name*"MP2 total energy: ",EMp2+EHF)
    t1 = print_time(EC,t1,"MP2",1)
    flush(stdout)
    if ecmethod.theory == "MP"
      continue
    end

    if ecmethod.exclevel[4] != :none
      error("no quadruples implemented yet...")
    end

    ecmethod_save = ecmethod
    if ecmethod.exclevel[3] in [:full, :pertiter]
      ecmethod = ECMethod("CCSD")
      ecmethod.unrestricted = ecmethod_save.unrestricted
    end
    ECC = calc_cc(EC, ecmethod)

    main_name = method_name(ecmethod)
    ecmethod = ecmethod_save # restore

    if closed_shell_method
      if ecmethod.exclevel[3] != :none
        do_full_t3 = (ecmethod.exclevel[3] == :full || ecmethod.exclevel[3] == :pertiter)
        save_pert_t3 = do_full_t3 && EC.options.cc.calc_t3_for_decomposition
        ET3, ET3b = calc_pertT(EC; save_t3 = save_pert_t3)
        println()
        println("$main_name[T] total energy: ",ECC+ET3b+EHF)
        println("$main_name(T) correlation energy: ",ECC+ET3)
        println("$main_name(T) total energy: ",ECC+ET3+EHF)
        if do_full_t3
          cc3 = (ecmethod.exclevel[3] == :pertiter)
          ECC = CoupledCluster.calc_ccsdt(EC, EC.options.cc.calc_t3_for_decomposition, cc3)
          main_name = method_name(ecmethod)
          println("$main_name correlation energy: ",ECC)
          println("$main_name total energy: ",ECC+EHF)
        end 
      end
    end
    println()
    flush(stdout)

    if ecmethod.theory[1:2] == "2D"
      W = load(EC,"2d_ccsd_W")[1]
      @printf "%26s %16.12f \n" "$main_name singlet energy:" EHF+ECC+W
      @printf "%26s %16.12f \n" "$main_name triplet energy:" EHF+ECC-W
      t1 = print_time(EC, t1,"CC",1)
      delete_temporary_files!(EC)
      draw_endline()
      return EHF, EMp2, ECC, W
    else
      println("$main_name correlation energy: ",ECC)
      println("$main_name total energy: ",ECC+EHF)
      t1 = print_time(EC, t1,"CC",1)
      delete_temporary_files!(EC)
      draw_endline()
      if length(method_names) == 1
        if ecmethod.exclevel[3] != :none
          return EHF, EMp2, ECC, ET3
        else
          return EHF, EMp2, ECC
        end
      end
    end
  end
end

end #module
