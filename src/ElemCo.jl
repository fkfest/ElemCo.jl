#!/usr/bin/env julia

"""
ELEctronic Methods of COrrelation 
"""

module ElemCo

include("abstractEC.jl")
include("utils.jl")
include("myio.jl")
include("mnpy.jl")
include("dump.jl")
include("integrals.jl")
include("msystem.jl")
include("diis.jl")

include("ecinfos.jl")
include("ecmethods.jl")
include("tensortools.jl")
include("fock.jl")
include("cc.jl")

include("bohf.jl")

include("dfhf.jl")
include("dfdump.jl")

try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using ArgParse
using .Utils
using .ECInfos
using .ECMethods
using .TensorTools
using .Focks
using .CoupledCluster
using .FciDump
using .MSystem
using .BOHF
using .DFHF
using .DfDump


export ECdriver 
export @ECsetup, @tryECsetup, @opt, @run, @dfhf, @dfints, @cc

""" 
  @ECsetup()

  setup `EC::ECInfo` from `geometry::String` and `basis::Dict{String,Any}` 
"""
macro ECsetup()
  return quote
    global $(esc(:EC)) = ECInfo(ms=MSys($(esc(:geometry)),$(esc(:basis))))
    setup!($(esc(:EC)))
  end
end

""" 
  @tryECsetup()

  setup `EC::ECInfo` from `geometry::String` and `basis::Dict{String,Any}` 
  if not already done 
"""
macro tryECsetup()
  return quote
    try
      $(esc(:EC)).ignore_error
    catch
      $(esc(:@ECsetup))
    end
  end
end

""" 
  @opt(what, kwargs...)

  set options for `EC::ECInfo`. If `EC` is not already setup, it will be done. 
  Usage: e.g., to set maxit=10 for scf: `@opt scf maxit=10` 
"""
macro opt(what, kwargs...)
  strwhat="$what"
  ekwa = [esc(a) for a in kwargs]
  return quote
    $(esc(:@tryECsetup))
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
    $(esc(:@tryECsetup))
    $method($(esc(:EC)); $(ekwa...))
  end
end

""" 
  @dfhf()

  run DFHF calculation and return MO coefficients (`ORBS`) and orbital energies (`EPS`)
"""
macro dfhf()
  return quote
    $(esc(:@tryECsetup))
    $(esc(:EPS)), $(esc(:ORBS)) = dfhf($(esc(:EC)))
  end
end

"""
  @dfints(orbs = nothing, fcidump = "")

  generate 2 and 4-idx MO integrals using density fitting.
  If `orbs` is given, the orbitals are used to generate the integrals, 
  otherwise the last orbitals (`ORBS`) are used.
  If `fcidump` is given, the integrals are written to the fcidump file.
"""
macro dfints(orbs = nothing, fcidump = "")
  return quote
    $(esc(:@tryECsetup))
    if isnothing($orbs)
      orbitals = $(esc(:ORBS))
    else
      orbitals = $orbs
    end
    dfdump($(esc(:EC)),orbitals, $fcidump)
  end
end

""" 
  @cc(method, kwargs...)

  run coupled cluster calculation
  The type of the method is determined by the first argument (ccsd/ccsd(t)/dcsd etc)
  The following keyword arguments are available:
  * `fcidump`: fcidump file (default: "", i.e., use integrals from `EC`)
  * `occa`: occupied α orbitals (default: "-")
  * `occb`: occupied β orbitals (default: "-")
"""
macro cc(method, kwargs...)
  strmethod="$method"
  ekwa = [esc(a) for a in kwargs]
  fcidump_given = false
  for a in ekwa
    if a.args[1].args[1] == :fcidump
      fcidump_given = true
    end
  end
  if fcidump_given
    return quote
      ECdriver($(esc(:EC)), $(esc(strmethod)); $(ekwa...))
    end
  else
    return quote
      ECdriver($(esc(:EC)), $(esc(strmethod)); fcidump="", $(ekwa...))
    end
  end
end

""" parse command line arguments """
function parse_commandline(EC::ECInfo)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--method", "-m"
      help = "method or list of methods to calculate"
      arg_type = String
      default = "dcsd"
    "--scratch", "-s"
      help = "scratch directory"
      arg_type = String
      default = "elemcojlscr"
    "--verbosity", "-v"
      help = "verbosity"
      arg_type = Int
      default = 2
    "--output", "-o"
      help = "output file"
      arg_type = String
      default = ""
    "--occa"
      help = "occupied α orbitals (in '1-3+5' format)"
      arg_type = String
      default = "-"
    "--occb"
      help = "occupied β orbitals (in '1-3+6' format)"
      arg_type = String
      default = "-"
    "--force", "-f"
      help = "supress some of the error messages (ignore_error)"
      action = :store_true
    "--choltol", "-c"
      help = "cholesky threshold"
      arg_type = Float64
      default = 1.e-6
    "--amptol", "-a"
      help = "amplitude threshold"
      arg_type = Float64
      default = 1.e-3
    "--save_t3"
      help = "save (T) for decomposition"
      action = :store_true
    "arg1"
      help = "input file (currently fcidump file)"
      default = "FCIDUMP"
    "--test", "-t"
      action = :store_true

  end
  args = parse_args(s)
  EC.scr = args["scratch"]
  EC.verbosity = args["verbosity"]
  EC.out = args["output"]
  EC.ignore_error = args["force"]
  EC.options.cholesky.thr = args["choltol"]
  EC.options.cc.ampsvdtol = args["amptol"]
  EC.options.cc.calc_t3_for_decomposition = args["save_t3"]
  fcidump_file = args["arg1"]
  method = args["method"]
  occa = args["occa"]
  occb = args["occb"]
  test = args["test"]
  if test
    include(joinpath(@__DIR__,"..","test","runtests.jl"))
    fcidump_file = ""
  end
  return fcidump_file, method, occa, occb
end

function run(method::String="ccsd", dumpfile::String="H2O.FCIDUMP", use_kext::Bool=true, occa="-", occb="-")
  EC = ECInfo()
  fcidump = joinpath(@__DIR__,"..","test",dumpfile)
  EC.options.cc.maxit = 100
  EC.options.cc.thr = 1.e-12
  EC.options.cc.use_kext = use_kext
  EC.options.cc.calc_d_vvvv = !use_kext
  EC.options.cc.calc_d_vvvo = !use_kext
  EC.options.cc.calc_d_vovv = !use_kext
  EC.options.cc.calc_d_vvoo = !use_kext
  EHF, EMP2, ECCSD = ECdriver(EC,method; fcidump, occa, occb)
  return ECCSD
end

function is_closed_shell(EC::ECInfo)
  SP = EC.space
  closed_shell = (SP['o'] == SP['O'] && !EC.fd.uhf)
  addname=""
  if !closed_shell
    addname = "U"
  end
  return closed_shell, addname
end

""" calculate fock matrix """
function calc_fock_matrix(EC::ECInfo, closed_shell)
  t1 = time_ns()
  if closed_shell
    EC.fock,EC.ϵo,EC.ϵv = gen_fock(EC)
    EC.fockb = EC.fock
    EC.ϵob = EC.ϵo
    EC.ϵvb = EC.ϵv
  else
    EC.fock,EC.ϵo,EC.ϵv = gen_fock(EC,SCα)
    EC.fockb,EC.ϵob,EC.ϵvb = gen_fock(EC,SCβ)
  end
  t1 = print_time(EC,t1,"fock matrix",1)
end

""" calculate HF energy """
function calc_HF_energy(EC::ECInfo, closed_shell)
  SP = EC.space
  if closed_shell
    EHF = sum(EC.ϵo) + sum(diag(integ1(EC.fd))[SP['o']]) + EC.fd.int0
  else
    EHF = 0.5*(sum(EC.ϵo)+sum(EC.ϵob) + sum(diag(integ1(EC.fd, SCα))[SP['o']]) + sum(diag(integ1(EC.fd, SCβ))[SP['O']])) + EC.fd.int0
  end
  return EHF
end

function ECdriver(EC::ECInfo, methods; fcidump="FCIDUMP", occa="-", occb="-")
  t1 = time_ns()
  method_names = split(methods)
  setup!(EC;fcidump,occa,occb)
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
    end
    # at the moment we always calculate MP2 first
    # calculate MP2
    if closed_shell_method
      EMp2, T2 = calc_MP2(EC)
    else
      EMp2, T2a, T2b, T2ab = calc_UMP2(EC)
    end
    println(add2name*"MP2 correlation energy: ",EMp2)
    println(add2name*"MP2 total energy: ",EMp2+EHF)
    t1 = print_time(EC,t1,"MP2",1)
    flush(stdout)

    if ecmethod.theory == "MP"
      continue
    end

    dc = (ecmethod.theory == "DC")

    if ecmethod.exclevel[4] != NoExc
      error("no quadruples implemented yet...")
    end

    if closed_shell_method
      if ecmethod.exclevel[1] == FullExc
        T1 = zeros(size(SP['v'],1),size(SP['o'],1))
      else
        T1 = zeros(0)
      end
      ECC, T1, T2 = calc_cc(EC, T1, T2, dc)
    else
      if ecmethod.exclevel[1] == FullExc
        T1a = zeros(size(SP['v'],1),size(SP['o'],1))
        T1b = zeros(size(SP['V'],1),size(SP['O'],1))
        if(!EC.options.cc.use_kext)
          error("open-shell CCSD only implemented with kext")
        end
      else
        T1a = zeros(0)
        T1b = zeros(0)
      end
      ECC, T1a, T1b, T2a, T2b, T2ab = calc_cc(EC,T1a,T1b,T2a,T2b,T2ab,dc)
    end

    if closed_shell_method
      main_name = method_name(T1,dc)
      if ecmethod.exclevel[3] != NoExc
        do_full_t3 = (ecmethod.exclevel[3] == FullExc || ecmethod.exclevel[3] == PertExcIter)
        save_pert_t3 = do_full_t3 && EC.options.cc.calc_t3_for_decomposition
        ET3, ET3b = calc_pertT(EC, T1, T2; save_t3 = save_pert_t3)
        println()
        println("$main_name[T] total energy: ",ECC+ET3b+EHF)
        println("$main_name(T) correlation energy: ",ECC+ET3)
        println("$main_name(T) total energy: ",ECC+ET3+EHF)
        if do_full_t3
          cc3 = (ecmethod.exclevel[3] == PertExcIter)
          ECC, T1, T2 = CoupledCluster.calc_ccsdt(EC, T1, T2, EC.options.cc.calc_t3_for_decomposition, cc3)
          if cc3
            main_name = "CC3"
          else
            main_name = "DC-CCSDT"
          end
          println("$main_name correlation energy: ",ECC)
          println("$main_name total energy: ",ECC+EHF)
        end 
      end
    else
      main_name = method_name(T1a,dc)
    end
    flush(stdout)

    println(add2name*"$main_name correlation energy: ",ECC)
    println(add2name*"$main_name total energy: ",ECC+EHF)
    t1 = print_time(EC, t1,"CC",1)
    if length(method_names) == 1
      if ecmethod.exclevel[3] != NoExc
        return EHF, EMp2, ECC, ET3
      else
        return EHF, EMp2, ECC
      end
    end
  end
end

function main()
  EC = ECInfo()
  fcidump, method_string, occa, occb = parse_commandline(EC)
  if fcidump == ""
    println("No input file given.")
    return
  end
  if EC.out != ""
    output = EC.out
  else
    output = nothing
  end
  redirect_stdio(stdout=output) do
    ECdriver(EC, method_string, fcidump=fcidump, occa=occa, occb=occb)
  end
end
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

end #module
