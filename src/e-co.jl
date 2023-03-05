#!/usr/bin/env julia

include("myio.jl")
include("mnpy.jl")
include("dump.jl")
include("diis.jl")
include("ecinfos.jl")
include("utils.jl")
include("ecmethods.jl")
include("tensortools.jl")
include("fock.jl")
include("cc.jl")

include("msystem.jl")
include("integrals.jl")

"""
Electron-Correlation methods 

"""
module eCo

using LinearAlgebra
try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
#BLAS.set_num_threads(1)
using ArgParse
using ..Utils
using ..ECInfos
using ..ECMethods
using ..TensorTools
using ..Focks
using ..CoupledCluster
using ..FciDump

function parse_commandline(EC::ECInfo)
  s = ArgParseSettings()
  @add_arg_table s begin
    "--method", "-m"
      help = "method or list of methods to calculate"
      arg_type = String
      default = "dcsd"
    "--scratch", "-s"
      help = "scratch directory"
      arg_type = String
      default = "e-cojlscr"
    "--verbosity", "-v"
      help = "verbosity"
      arg_type = Int
      default = 2
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
    "arg1"
      help = "input file (currently fcidump file)"
      default = "FCIDUMP"
  end
  args = parse_args(s)
  EC.scr = args["scratch"]
  EC.verbosity = args["verbosity"]
  EC.ignore_error = args["force"]
  fcidump_file = args["arg1"]
  method = args["method"]
  occa = args["occa"]
  occb = args["occb"]
  return fcidump_file, method, occa, occb
end

function main()
  t1 = time_ns()
  EC = ECInfo()
  fcidump, method_string, occa, occb = parse_commandline(EC)
  method_names = split(method_string)
  # create scratch directory
  mkpath(EC.scr)
  EC.scr = mktempdir(EC.scr)
  # read fcidump intergrals
  EC.fd = read_fcidump(fcidump)
  t1 = print_time(EC,t1,"read fcidump",1)
  println(size(EC.fd.int2))
  norb = headvar(EC.fd, "NORB")
  nelec = headvar(EC.fd, "NELEC")

  EC.space['o'], EC.space['v'], EC.space['O'], EC.space['V'] = get_occvirt(EC, occa, occb, norb, nelec)
  EC.space[':'] = 1:headvar(EC.fd,"NORB")

  SP(sp::Char) = EC.space[sp]

  closed_shell = (EC.space['o'] == EC.space['O'] && !EC.fd.uhf)

  addname=""
  if !closed_shell
    addname = "U"
  end

  # calculate fock matrix 
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

  # calculate HF energy
  if closed_shell
    EHF = sum(EC.ϵo) + sum(diag(integ1(EC.fd))[SP('o')]) + EC.fd.int0
  else
    EHF = 0.5*(sum(EC.ϵo)+sum(EC.ϵob) + sum(diag(integ1(EC.fd, SCα))[SP('o')]) + sum(diag(integ1(EC.fd, SCβ))[SP('O')])) + EC.fd.int0
  end
  println(addname*"HF energy: ",EHF)

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

    if ecmethod.theory == "MP"
      continue
    end
    closed_shell || error("Open-shell methods not implemented yet")
    dc = (ecmethod.theory == "DC")
    T1 = nothing
    if ecmethod.exclevel[1] == FullExc
      T1 = zeros(size(SP('v'),1),size(SP('o'),1))
    end
    if ecmethod.exclevel[3] != NoExc
      error("no triples implemented yet...")
    end
    ECC = calc_cc!(EC, T1, T2, dc)
    println("$mname correlation energy: ",ECC)
    println("$mname total energy: ",ECC+EHF)
    t1 = print_time(EC, t1,"CC",1)
  end
end
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

end #module
