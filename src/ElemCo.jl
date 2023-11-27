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

include("dfmcscf.jl")

try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using ArgParse
using TimerOutputs, TensorOperations, BenchmarkTools
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
using .DFMCSCF
using .DfDump


export ECdriver 
export @ECsetup, @tryECsetup, @opt, @run, @dfhf, @dfints, @cc

""" 
    @ECsetup()

  Setup `EC::ECInfo` from variables `geometry::String` and `basis::Dict{String,Any}`.

  # Examples
```jldoctest
geometry="\nHe 0.0 0.0 0.0"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mp2fit"=>"cc-pvdz-rifit")
@ECsetup
# output
Occupied orbitals:[1]
1
```
"""
macro ECsetup()
  return quote
    global $(esc(:EC)) = ECInfo(ms=MSys($(esc(:geometry)),$(esc(:basis))))
    setup!($(esc(:EC)))
  end
end

""" 
    @tryECsetup()

  Setup `EC::ECInfo` from `geometry::String` and `basis::Dict{String,Any}` 
  if not already done.
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

  Set options for `EC::ECInfo`. 
    
  If `EC` is not already setup, it will be done. 

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

  Run DFHF calculation and return MO coefficients (`ORBS`) and orbital energies (`EPS`).
"""
macro dfhf()
  return quote
    $(esc(:@tryECsetup))
    $(esc(:EPS)), $(esc(:ORBS)) = dfhf($(esc(:EC)))
  end
end

"""
    @dfints(orbs = nothing, fcidump = "")

  Generate 2 and 4-idx MO integrals using density fitting.

  If `orbs::Matrix` is given, the orbitals are used to generate the integrals, 
  otherwise the last orbitals (`ORBS`) are used.
  If `fcidump::String` is given, the integrals are written to the fcidump file.
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

""" 
    parse_commandline(EC::ECInfo)

Parse command line arguments. 
"""
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

function run_mcscf()
  xyz="angstrom
     Fe  -1.6827629   -0.5620638    0.0006858
     N   -1.7654701    1.7386430   -0.0155915
     H   -2.3091690    2.1062136    0.7643967
     H   -0.8575605    2.1961896    0.0569999
     H   -2.1985074    2.1444062   -0.8446345"

    #  N    0.6370910   -0.4829211    0.0198374
    #  H    1.0410909   -0.1381987    0.8898340
    #  H    1.0933299   -1.3794527   -0.1444124
    #  H    1.0148101    0.1305669   -0.7014791

    #  N   -4.0024708   -0.5711646   -0.0821798
    #  H   -4.4120209   -1.4547568   -0.3829660
    #  H   -4.4588691   -0.3582133    0.8039661
    #  H   -4.3743783    0.1202099   -0.7325748
    #  N   -1.6769265   -2.8574940    0.0683045
    #  H   -2.4248044   -3.2491480    0.6399273
    #  H   -1.8046273   -3.2785580   -0.8511997
    #  H   -0.8224786   -3.2773553    0.4326991
    #  N   -1.6947644   -0.6233020    2.2950899
    #  H   -0.9567299   -1.2059553    2.6891561
    #  H   -1.5535805    0.2946430    2.7152532
    #  H   -2.5558826   -0.9737822    2.7131764
    #  N   -1.5929695   -0.5620411   -2.2999504
    #  H   -1.0185358   -1.3236265   -2.6592866
    #  H   -2.4963312   -0.6738135   -2.7585987
    #  H   -1.1916622    0.2796750   -2.7122128

  basis = Dict("ao"=>"cc-pVTZ",
             "jkfit"=>"cc-pvtz-rifit",
             "mp2fit"=>"cc-pvtz-rifit")

  EC = ECInfo(ms=MSys(xyz,basis))
  setup!(EC,ms2=4,charge=2)
  to = TimerOutputs.get_defaulttimer()
  TimerOutputs.reset_timer!(to)
  E,cMO =  dfmcscf(EC,direct=false, IterMax=300)
  display(to)
  TimerOutputs.reset_timer!(to)
end

function run_mcscf_s(IterMax::Number)
  xyz="bohr
      O      0.000000000    0.000000000   -0.130186067
      H1     0.000000000    1.489124508    1.033245507
      H2     0.000000000   -1.489124508    1.033245507"


  basis = Dict("ao"=>"cc-pVDZ",
              "jkfit"=>"cc-pvtz-jkfit",
              "mp2fit"=>"cc-pvdz-rifit")

  EC = ECInfo(ms=MSys(xyz,basis))
  @timeit "setup" setup!(EC,ms2=2,charge=-2)
  @timeit "dfmcscf" E,cMO = dfmcscf(EC,direct=false,IterMax=IterMax)
  
end

function run_calcH()
  to = TimerOutputs.get_defaulttimer()
  TimerOutputs.reset_timer!(to)
  xyz="bohr
      O      0.000000000    0.000000000   -0.130186067
      H1     0.000000000    1.489124508    1.033245507
      H2     0.000000000   -1.489124508    1.033245507"


  basis = Dict("ao"=>"cc-pVDZ",
              "jkfit"=>"cc-pvtz-jkfit",
              "mp2fit"=>"cc-pvdz-rifit")

  EC = ECInfo(ms=MSys(xyz,basis))
  setup!(EC,ms2=2,charge=-2)

  @timeit "set up" begin
    cMO = rand(24,24)
    D1 = 1.0 * Matrix(I, 2, 2)
    @tensoropt D2[t,u,v,w] := D1[t,u]*D1[v,w] - D1[t,w]*D1[v,u]
    fock = rand(24,24)
    fockClosed = rand(24,24)
    A = rand(24,24)
    # @btime h = calc_h($EC, $cMO, $D1, $D2, $fock, $fockClosed, $A)

    # @timeit "setup" begin
    occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
    occ1o = setdiff(EC.space['o'],occ2)
    occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
    n_1o = size(occ1o, 1)
    n_2 = size(occ2,1)
    n_v = size(occv,1)
    n_occ = n_2+n_1o
    n_open = n_1o+n_v
    n_MO = size(cMO,2)
    μνL = rand(24,24,139)
    μjL = rand(24,5,139)
    μuL = rand(24,2,139)
    μνL = 0
    G = zeros((n_MO,n_MO,n_occ,n_occ))
  end
  μνL = rand(24,24,139)


  @timeit "Gij" begin
    # Gij
    @timeit "Gij_1" @tensoropt pjL[p,j,L] := μjL[μ,j,L] * cMO[μ,p] # to transfer the first index from atomic basis to molecular basis
    @timeit "Gij_2" @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] = 8 * pjL[r,i,L] * pjL[s,j,L]
    # @btime @tensoropt G[:,:,1:$n_2,1:$n_2][r,s,i,j] = 8 * $pjL[r,i,L] * $pjL[s,j,L]
    @timeit "Gij_3" @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * pjL[s,i,L] * pjL[r,j,L]
    @timeit "Gij_4" ijL = pjL[occ2,:,:]
    @timeit "Gij_5" @tensoropt pνL[p,ν,L] := μνL[μ,ν,L] * cMO[μ,p] 
    @timeit "Gij_9" @tensoropt pqL[p,q,L] := pνL[p,ν,L] * cMO[ν,q]
    @timeit "Gij_6" @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * ijL[i,j,L] * pqL[r,s,L]
    @timeit "Gij_7" Iij = 1.0 * Matrix(I, length(occ2), length(occ2))
    @timeit "Gij_8" @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] += 2 * fock[μ,ν] * cMO[μ,r] * cMO[ν,s] * Iij[i,j] 
    
  end

  @timeit "Gtj" begin
    # Gtj
    @timeit "Gtj_1" @tensoropt puL[p,u,L] := μuL[μ,u,L] * cMO[μ,p] #transfer from atomic basis to molecular basis
    @timeit "Gtj_2" @tensoropt testStuff[r,s,v,j] := puL[r,v,L] * pjL[s,j,L]
    @timeit "Gtj_3" @tensoropt multiplier[r,s,v,j] := 4 * puL[r,v,L] * pjL[s,j,L]
    @timeit "Gtj_4" @tensoropt multiplier[r,s,v,j] -= puL[s,v,L] * pjL[r,j,L]
    @timeit "Gtj_5" tjL = pjL[occ1o,:,:]
    @timeit "Gtj_6" @tensoropt multiplier[r,s,v,j] -= pqL[r,s,L] * tjL[v,j,L]
    @timeit "Gtj_7" @tensoropt G[:,:,n_2+1:n_occ,1:n_2][r,s,t,j] = multiplier[r,s,v,j] * D1[t,v]
  end

  @timeit "Gtu" begin
    # Gtu 
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] = fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s] * D1[t,u]
    tuL = pqL[occ1o, occ1o, :]
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += pqL[r,s,L] * (tuL[v,w,L] * D2[t,u,v,w])
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += 2 * (puL[r,v,L] * puL[s,w,L]) * D2[t,v,u,w]
  end

  # Gjt
  @timeit "Gjt" G[:,:,1:n_2,n_2+1:n_occ] = permutedims(G[:,:,n_2+1:n_occ,1:n_2], [2,1,4,3])

  @timeit "Other stuff" begin
    if findmax(occ2)[1] > findmin(occ1o)[1] || findmax(occ1o)[1] > findmin(occv)[1]
      println("G reordered!")
      G = G[[occ2;occ1o;occv];[occ2;occ1o;occv];:;:]
    end

    # calc h with G 
    I_kl = 1.0 * Matrix(I, n_2+n_1o, n_2+n_1o)
    h = zeros((n_open,n_occ,n_open,n_occ))
    A = A[:,1:n_occ]
    @tensoropt h[r,k,s,l] += 2 * G[n_2+1:end,n_2+1:end,:,:][r,s,k,l]
    @tensoropt h[1:n_1o,:,:,:][r,k,s,l] -= 2 * G[1:n_occ,n_2+1:end,n_2+1:end,:][k,s,r,l]
    @tensoropt h[:,:,1:n_1o,:][r,k,s,l] -= 2 * G[n_2+1:end,1:n_occ,:,n_2+1:end][r,l,k,s]
    @tensoropt h[1:n_1o,:,1:n_1o,:][r,k,s,l] += 2 * G[1:n_occ,1:n_occ,n_2+1:end,n_2+1:end][k,l,r,s]
    for i in 1:n_occ
      h[:,i,1:n_1o,i] -= A[n_2+1:end,n_2+1:end]
      h[1:n_1o,i,:,i] -= transpose(A)[n_2+1:end,n_2+1:end]
    end
    for i in 1:n_open
      h[i,:,i,:] -= A[1:n_occ,:]
      h[i,:,i,:] -= transpose(A)[:,1:n_occ]
    end
    for i in 1:n_1o
      h[i,:,1:n_1o,n_2+i] += A[1:n_occ,n_2+1:end]
      h[i,:,:,n_2+i] += transpose(A)[:,n_2+1:end]
      h[:,n_2+i,i,:] += A[n_2+1:end,:]
      h[1:n_1o,n_2+i,i,:] += transpose(A)[n_2+1:end,1:n_occ]
    end

    d = n_occ * n_open
    h = reshape(h, d, d)
  end
  display(to)
  TimerOutputs.reset_timer!(to)
  r = " "
end

function run_calcfock()
  to = TimerOutputs.get_defaulttimer()
  TimerOutputs.reset_timer!(to)
  
  # xyz="bohr
  #   O      0.000000000    0.000000000   -0.130186067
  #   H1     0.000000000    1.489124508    1.033245507
  #   H2     0.000000000   -1.489124508    1.033245507"
  xyz="angstrom
    Fe  -1.6827629   -0.5620638    0.0006858
    N   -1.7654701    1.7386430   -0.0155915
    H   -2.3091690    2.1062136    0.7643967
    H   -0.8575605    2.1961896    0.0569999
    H   -2.1985074    2.1444062   -0.8446345"

  basis = Dict("ao"=>"cc-pVTZ",
             "jkfit"=>"cc-pvtz-rifit",
             "mp2fit"=>"cc-pvtz-rifit")

  EC = ECInfo(ms=MSys(xyz,basis))
  setup!(EC,ms2=4,charge=2)


  direct = false
  Enuc = generate_integrals(EC; save3idx=!direct)
  cMO = rand(140,140)
  D1 = 1.0 * Matrix(I, 4, 4)
  @tensoropt D2[t,u,v,w] := D1[t,u]*D1[v,w] - D1[t,w]*D1[v,u]

  @timeit "setup" begin
    @timeit "setup1" occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
    @timeit "setup2" occ1o = setdiff(EC.space['o'],occ2)
    @timeit "setup3" CMO2 = cMO[:,occ2]
    @timeit "setup4" CMOa = cMO[:,occ1o] # to be modified
    @timeit "load" μνL = load(EC,"munuL")
    # @timeit "setup6" μjL = load(EC,"mudL")
    # @timeit "setup7" μuL = load(EC,"muaL")
    #μνL = rand(24,24,139)
    μjL = rand(140,15,415)
    μuL = rand(140,4,415)
  end

  # fockClosed
  @timeit "fockClosed" begin
    @timeit "fockClosed1" hsmall = load(EC,"hsmall")
    @timeit "fockClosed2" @tensoropt L[L] := μjL[μ,j,L] * CMO2[μ,j]
    @timeit "fockClosed3" @tensoropt fockClosed[μ,ν] := hsmall[μ,ν] - μjL[μ,j,L]*μjL[ν,j,L]
    @timeit "fockClosed4" @tensoropt fockClosed[μ,ν] += 2.0*L[L]*μνL[μ,ν,L]
    hsmall = 0
  end

  # fock
  @timeit "fock" begin
    @timeit "fock1" fock =  deepcopy(fockClosed)
    # @timeit "fock2" @tensoropt μuLD[μ,t,L] := μuL[μ,u,L] * D1[t,u]
    # @timeit "fock3" @tensoropt fock[μ,ν] -= 0.5 * μuLD[μ,t,L] * μuL[ν,t,L]
    # @timeit "fock4" @tensoropt LD[L] := μuLD[μ,t,L] * CMOa[μ,t]
    @timeit "focktest" @tensoropt fock[μ,ν] -= 0.5 * μuL[μ,u,L] * D1[t,u] * μuL[ν,t,L]
    @timeit "fock4" @tensoropt LD[L] := μuL[μ,u,L] * D1[t,u] * CMOa[μ,t]
    @timeit "fock5" @tensoropt fock[μ,ν] += LD[L] * μνL[μ,ν,L]
  end

  @timeit "A" begin
    # Apj
    @timeit "A1" @tensoropt Apj[p,j] := 2 * (fock[μ,ν] * CMO2[ν,j]) * cMO[μ,p]
    # Apu
    @timeit "A2" @tensoropt Apu[p,u] := ((fockClosed[μ,ν] * CMOa[ν,v]) * cMO[μ,p]) * D1[v,u]
    @timeit "A3" @tensoropt Apu[p,u] += (((μuL[ν,v,L] * CMOa[ν,w]) * D2[t,u,v,w]) * μuL[μ,t,L]) * cMO[μ,p]
    @timeit "A4" A = zeros((size(cMO,2),size(cMO,2)))
    @timeit "A5" A[:,occ2] = Apj
    @timeit "A6" A[:,occ1o] = Apu # to be modified
  end

  @timeit "g" begin
    @timeit "g1" @tensoropt g[r,s] := A[r,s] - A[s,r]
    @timeit "g2" occv = setdiff(1:size(A,1), EC.space['o']) # to be modified
    @timeit "g3" grk = g[[occ1o;occv],[occ2;occ1o]] # to be modified
    @timeit "g4" grk = reshape(grk, size(grk,1) * size(grk,2))
  end

  display(to)
  TimerOutputs.reset_timer!(to)
  return fock, fockClosed
end

 function run(method::String="ccsd", dumpfile::String="H2O.FCIDUMP", occa="-", occb="-", use_kext::Bool=true)
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

""" 
    calc_HF_energy(EC::ECInfo, closed_shell)

  Calculate HF energy from FCIDump and EC info. 
"""
function calc_HF_energy(EC::ECInfo, closed_shell)
  SP = EC.space
  if closed_shell
    EHF = sum(EC.ϵo) + sum(diag(integ1(EC.fd))[SP['o']]) + EC.fd.int0
  else
    EHF = 0.5*(sum(EC.ϵo)+sum(EC.ϵob) + sum(diag(integ1(EC.fd, SCα))[SP['o']]) + sum(diag(integ1(EC.fd, SCβ))[SP['O']])) + EC.fd.int0
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
