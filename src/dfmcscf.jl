module DFMCSCF
using LinearAlgebra, TensorOperations, Printf, TimerOutputs
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.DIIS
using ..ElemCo.TensorTools
using ..ElemCo.DFHF

export dfmcscf, davidson, calc_h
export InitialVectorType, RANDOM, INHERIT, GRADIENT_SET, GRADIENT_SETPLUS
export HessianType, SO, SCI, SO_SCI

"""
    dfmcs(EC::ECInfo, cMO::Matrix)

  DF-MCSCF calculation.
"""

"""
    Type of initial guess vectors of Davidson iterations

  Possible values:
  - RANDOM: one random vector
  - INHERIT: from last macro/micro iterations
  - GRADIENT_SET: b0 as [1,0,0,...], b1 as gradient
  - GRADIENT_SETPLUS: b0, b1 as GRADIENT_SET, b2 as zeros but 1 at the first closed-virtual rotation parameter
"""
@enum InitialVectorType RANDOM INHERIT GRADIENT_SET GRADIENT_SETPLUS

"""
    Type of Hessian matrix

  Possible values:
  - SO: Second Order Approximation
  - SCI: Super CI
  - SO-SCI: Second Order Approximation combing Super CI
"""

@enum HessianType SO SCI SO_SCI

"""
    denMatCreate(EC::ECInfo)

Calculate the one particle density matrix and two particle density matrix of active electrons
for high-spin determinant.
D1[t,u] = ``^1D^t_u = ⟨Ψ|\\hat{E}^t_u|Ψ⟩  = ⟨ Ψ |∑_σ \\hat{a}^†_{tσ} \\hat{a}_{uσ}|Ψ⟩``, 
D2[t,u,v,w] = ``=^2D^{tv}_{uw}=0.5 ⟨Ψ|\\hat{E}^{tv}_{uw}+\\hat{E}^{uv}_{tw}|Ψ⟩``, 
in which ``\\hat{E}_{tu,vw} = \\hat{E}^t_u \\hat{E}^v_w - δ_{uv} \\hat{E}^t_w = ∑_{στ}\\hat{a}^†_{tσ} \\hat{a}^†_{vτ} \\hat{a}_{wτ} \\hat{a}_{uσ}``.
Return D1 and D2.
"""
function denMatCreate(EC::ECInfo)
  SP = EC.space
  nact = length(SP['o'])- length(SP['O']) # to be modified
  D1 = 1.0 * Matrix(I, nact, nact)
  @tensoropt D2[t,u,v,w] := (D1[t,u]*D1[v,w] - D1[t,w]*D1[v,u]) * 0.5
  @tensoropt D2[t,u,v,w] += (D1[u,t]*D1[v,w] - D1[u,w]*D1[v,t]) * 0.5
  return D1, D2
end

"""
    dffockCAS(EC::ECInfo, cMO::Matrix, D1::Matrix)

Calculate fock matrices in atomic orbital basis.     
Return matrix fock and fockClosed.
fockClosed[μ,ν] = ``^cf_μ^ν = h_μ^ν + 2v_{μi}^{νi} - v_{μi}^{iν}``, 
fock[μ,ν] = ``f_μ^ν = ^cf_μ^ν + D^t_u (v_{μt}^{νu} - 0.5 v_{μt}^{uν})``.
"""
function dffockCAS(EC::ECInfo, cMO::Matrix, D1::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  @timeit "loadμνL" μνL = load(EC,"munuL")
  @tensoropt μjL[μ,j,L] := μνL[μ,ν,L] * CMO2[ν,j]
  save(EC,"mudL",μjL)
  @tensoropt μuL[μ,u,L] := μνL[μ,ν,L] * CMOa[ν,u]
  save(EC,"muaL",μuL)
  @tensoropt abL[a,b,L] := μνL[μ,ν,L] * cMO[:,occv][μ,a] * cMO[:,occv][ν,b]
  save(EC,"abL", abL)

  # fockClosed
  hsmall = load(EC,"hsmall")
  @tensoropt L[L] := μjL[μ,j,L] * CMO2[μ,j]
  @tensoropt fockClosed[μ,ν] := hsmall[μ,ν] - μjL[μ,j,L]*μjL[ν,j,L]
  @tensoropt fockClosed[μ,ν] += 2.0*L[L]*μνL[μ,ν,L]

  # fock
  fock =  deepcopy(fockClosed)
  @tensoropt μuLD[μ,t,L] := μuL[μ,u,L] * D1[t,u]
  @tensoropt fock[μ,ν] -= 0.5 * μuLD[μ,t,L] * μuL[ν,t,L]
  @tensoropt LD[L] := μuLD[μ,t,L] * CMOa[μ,t]
  @tensoropt fock[μ,ν] += LD[L] * μνL[μ,ν,L]
  return fock, fockClosed
end

"""
    dfACAS(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix)

Calculate the A-intermediate matrix in molecular orbital basis.
return matrix A[p,q]
"""
function dfACAS(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  μuL = load(EC,"muaL")
  # Apj
  @tensoropt Apj[p,j] := 2 * (fock[μ,ν] * CMO2[ν,j]) * cMO[μ,p]
  # Apu
  @tensoropt Apu[p,u] := ((fockClosed[μ,ν] * CMOa[ν,v]) * cMO[μ,p]) * D1[v,u]
  @tensoropt Apu[p,u] += (((μuL[ν,v,L] * CMOa[ν,w]) * D2[t,u,v,w]) * μuL[μ,t,L]) * cMO[μ,p]
  A = zeros((size(cMO,2),size(cMO,2)))
  A[:,occ2] = Apj
  A[:,occ1o] = Apu # to be modified
  return A
end

"""
    calc_g(A::Matrix, EC::ECInfo)

Calculate the orbital gradient g by antisymmetrizing the matrix A
and rearranging the elements.
The order of the elements in g[r,k] is (active|virtual) × (closed-shell|active) 
"""
function calc_g(A::Matrix, EC::ECInfo)
  @tensoropt g[r,s] := A[r,s] - A[s,r]
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:size(A,1), EC.space['o']) # to be modified
  n_1o = size(occ1o, 1)
  n_2 = size(occ2,1)
  n_v = size(occv,1)
  g21 =reshape(g[occ1o,occ2], n_1o * n_2)
  g31 =reshape(g[occv,occ2], n_v * n_2)
  g22 =reshape(g[occ1o,occ1o], n_1o * n_1o)
  g32 =reshape(g[occv,occ1o], n_v * n_1o)
  g_blockwise = [g21;g31;g22;g32]
  g_blockwise .= g_blockwise .* 2.0
  return g_blockwise
end

"""
    calc_h(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix, A::Matrix)

Calculate Hessian matrix `h[rk,sl]`. `rk` and `sl` are combined indices of `r,k` and `s,l`, where
indexes r,s refer to open orbitals reordered as (active|virtual), 
and indexes k,l refer to occupied orbitals reordered as (closed-shell|active).
"""
function calc_h(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix, A::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
  n_1o = size(occ1o, 1)
  n_2 = size(occ2,1)
  n_v = size(occv,1)
  n_occ = n_2+n_1o
  n_open = n_1o+n_v
  n_MO = size(cMO,2)
  μνL = load(EC,"munuL")
  μjL = load(EC,"mudL")
  μuL = load(EC,"muaL")
  G = zeros((n_MO,n_MO,n_occ,n_occ))

  # Gij
  @timeit "Gij" begin
    @tensoropt pjL[p,j,L] := μjL[μ,j,L] * cMO[μ,p] # to transfer the first index from atomic basis to molecular basis
    @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] = 8 * pjL[r,i,L] * pjL[s,j,L]
    @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * pjL[s,i,L] * pjL[r,j,L]
    ijL = pjL[occ2,:,:]
    @tensoropt pqL[p,q,L] := μνL[μ,ν,L] * cMO[μ,p] * cMO[ν,q]
    @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * ijL[i,j,L] * pqL[r,s,L]
    Iij = 1.0 * Matrix(I, length(occ2), length(occ2))
    @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] += 2 * fock[μ,ν] * cMO[μ,r] * cMO[ν,s] * Iij[i,j] 
  end
  # Gtj
  @timeit "Gtj" begin
    @tensoropt puL[p,u,L] := μuL[μ,u,L] * cMO[μ,p] #transfer from atomic basis to molecular basis
    @tensoropt multiplier[r,s,v,j] := 4 * puL[r,v,L] * pjL[s,j,L]
    @tensoropt multiplier[r,s,v,j] -= puL[s,v,L] * pjL[r,j,L]
    tjL = pjL[occ1o,:,:]
    @tensoropt multiplier[r,s,v,j] -= pqL[r,s,L] * tjL[v,j,L]
    @tensoropt G[:,:,n_2+1:n_occ,1:n_2][r,s,t,j] = multiplier[r,s,v,j] * D1[t,v]
  end

  # Gtu 
  @timeit "Gtu" begin
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] = fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s] * D1[t,u]
    tuL = pqL[occ1o, occ1o, :]
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += pqL[r,s,L] * (tuL[v,w,L] * D2[t,u,v,w])
    @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += 2 * (puL[r,v,L] * puL[s,w,L]) * D2[t,v,u,w]
  end

  # Gjt
  G[:,:,1:n_2,n_2+1:n_occ] = permutedims(G[:,:,n_2+1:n_occ,1:n_2], [2,1,4,3])


  if findmax(occ2)[1] > findmin(occ1o)[1] || findmax(occ1o)[1] > findmin(occv)[1]
    println("G reordered!")
    G = G[[occ2;occ1o;occv];[occ2;occ1o;occv];:;:]
  end

  # calc h with G 
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
  return h
end

function calc_h_SO(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix, A::Matrix, HT::HessianType = SO)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  num_MO = [n_2,n_1o,n_v]
  index_MO = [occ2,occ1o,occv]
  μjL = load(EC,"mudL")
  μuL = load(EC,"muaL")
  if HT == SO
    abL = load(EC,"abL")
  else
    abL = 0
  end
  @tensoropt fock_MO[r,s] := fock[μ,ν] * cMO[μ,r] * cMO[ν,s]
  @tensoropt fockClosed_MO[r,s] := fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s]
  A = A + A'

  # precalculate the density fitting integrals in molecular orbital basis
  @tensoropt ijL[i,j,L] := μjL[μ,j,L] * cMO[:,occ2][μ,i]
  @tensoropt tiL[t,i,L] := μjL[μ,i,L] * cMO[:,occ1o][μ,t]
  @tensoropt aiL[a,i,L] := μjL[μ,i,L] * cMO[:,occv][μ,a]
  @tensoropt tuL[t,u,L] := μuL[μ,u,L] * cMO[:,occ1o][μ,t]
  @tensoropt atL[a,t,L] := μuL[μ,t,L] * cMO[:,occv][μ,a]

  DFint_MO = [[ijL,tiL,aiL],[tiL,tuL,atL],[aiL,atL,abL]]
  Iij = 1.0 * Matrix(I,n_2,n_2)
  Itu = 1.0 * Matrix(I,n_1o,n_1o)
  Iab = 1.0 * Matrix(I,n_v,n_v)

  function G_risj_calc(typer::Integer, types::Integer)
    G_risj = zeros(num_MO[typer],n_2,num_MO[types],n_2)
    fock_rs = fock_MO[index_MO[typer], index_MO[types]]
    riL = DFint_MO[typer][1] #2,3
    sjL = DFint_MO[types][1] #2,3
    rsL = DFint_MO[typer][types] #22,32,33
    ijL = DFint_MO[1][1]
    @tensoropt G_risj[r,i,s,j] += fock_rs[r,s] * Iij[i,j] * 2.0
    @tensoropt G_risj[r,i,s,j] += riL[r,i,L] * sjL[s,j,L] * 8.0
    @tensoropt G_risj[r,i,s,j] -= sjL[s,i,L] * riL[r,j,L] * 2.0
    @tensoropt G_risj[r,i,s,j] -= rsL[r,s,L] * ijL[i,j,L] * 2.0
    fock_rs = 0
    return G_risj
  end

  function G_rtsj_calc(typer::Integer,types::Integer)
    G_rvsj = zeros(num_MO[typer],n_1o,num_MO[types],n_2)
    rvL = DFint_MO[typer][2] #1,2,3 might need reverse
    sjL = DFint_MO[types][1] #2,3
    svL = DFint_MO[types][2] #2,3
    rjL = DFint_MO[typer][1] #1,2,3
    rsL = DFint_MO[typer][types] #12,13,22,23,32,33 might need reverse
    vjL = DFint_MO[2][1]
    if typer < 2
      @tensoropt G_rvsj[r,v,s,j] += rvL[v,r,L] * sjL[s,j,L] * 4.0
    else
      @tensoropt G_rvsj[r,v,s,j] += rvL[r,v,L] * sjL[s,j,L] * 4.0
    end
    @tensoropt G_rvsj[r,v,s,j] -= svL[s,v,L] * rjL[r,j,L]
    if typer < types
      @tensoropt G_rvsj[r,v,s,j] -= rsL[s,r,L] * vjL[v,j,L]
    else
      @tensoropt G_rvsj[r,v,s,j] -= rsL[r,s,L] * vjL[v,j,L]
    end
    @tensoropt G_rtsj[r,t,s,j] := D1[t,v] * G_rvsj[r,v,s,j]
    G_rvsj = 0 
    return G_rtsj
  end

  function G_rtsu_calc(typer::Integer,types::Integer)
    G_rtsu = zeros(num_MO[typer],n_1o,num_MO[types],n_1o)
    fockClosed_rs = fockClosed_MO[index_MO[typer], index_MO[types]]
    rsL = DFint_MO[typer][types] #11,21,22,31,32,33
    vwL = DFint_MO[2][2]
    swL = DFint_MO[types][2] #1,2,3 might need reverse
    rvL = DFint_MO[typer][2] #1,2,3 might need reverse
    @tensoropt G_rtsu[r,t,s,u] += fockClosed_rs[r,s] * D1[t,u]
    @tensoropt G_rtsu[r,t,s,u] += rsL[r,s,L] * vwL[v,w,L] * D2[t,u,v,w]
    if types < 2
      if typer < 2
        @tensoropt G_rtsu[r,t,s,u] += rvL[v,r,L] * swL[w,s,L] * D2[t,v,u,w] * 2.0
      else
        @tensoropt G_rtsu[r,t,s,u] += rvL[r,v,L] * swL[w,s,L] * D2[t,v,u,w] * 2.0
      end
    else
      if typer < 2
        @tensoropt G_rtsu[r,t,s,u] += rvL[v,r,L] * swL[s,w,L] * D2[t,v,u,w] * 2.0
      else
        @tensoropt G_rtsu[r,t,s,u] += rvL[r,v,L] * swL[s,w,L] * D2[t,v,u,w] * 2.0
      end
    end
    fockClosed_rs = 0
    return G_rtsu
  end

  # h_3131 ==> G3131 needed, the largest and most memory consuming part
  h_3131 = G_risj_calc(3,3)
  h_3131 .= h_3131 .* 2.0
  @tensoropt h_3131[a,i,b,j] -= Iab[a,b] * A[occ2,occ2][i,j]

  # h_2121 --> G2121, G1221, G1212 needed
  h_2121 = G_risj_calc(2,2)
  h_2121 .= h_2121 .* 2.0
  @tensoropt h_2121[t,i,u,j] -= Iij[i,j] * A[occ1o,occ1o][t,u]
  G_1221 = G_rtsj_calc(1,2)
  @tensoropt h_2121[t,i,u,j] -= G_1221[i,t,u,j] * 2.0
  @tensoropt h_2121[t,i,u,j] -= G_1221[j,u,t,i] * 2.0
  G_1212 = G_rtsu_calc(1,1)
  @tensoropt h_2121[t,i,u,j] += G_1212[i,t,j,u] * 2.0
  @tensoropt h_2121[t,i,u,j] -= Itu[t,u] * A[occ2,occ2][i,j]
  G_1221 = 0
  G_1212 = 0

  # h_3121 --> G3121, G1231 needed
  h_3121 = G_risj_calc(3,2)
  h_3121 .= h_3121 .* 2.0
  @tensoropt h_3121[a,i,t,j] -= Iij[i,j] * A[occv,occ1o][a,t]
  G_1231 = G_rtsj_calc(1,3)
  @tensoropt h_3121[a,i,t,j] -= G_1231[j,t,a,i] * 2.0

  # h_2221 --> G2221, G2212 each for twice
  G_2221 = G_rtsj_calc(2,2)
  @tensoropt h_2221[t,u,v,i] := G_2221[t,u,v,i] * 2.0 - G_2221[u,t,v,i] * 2.0
  G_2212 = G_rtsu_calc(2,1)
  @tensoropt h_2221[t,u,v,i] += -2.0 * G_2212[t,u,i,v] + 2.0 * G_2212[u,t,i,v]
  @tensoropt h_2221[t,u,v,i] += Itu[u,v] * A[occ1o,occ2][t,i]
  @tensoropt h_2221[t,u,v,i] -= Itu[t,v] * A[occ1o,occ2][u,i]
  G_2221 = 0
  G_2212 = 0

  # h_2231 --> G2231 twice
  G_2231 = G_rtsj_calc(2,3)
  @tensoropt h_2231[t,u,a,i] := G_2231[t,u,a,i] * 2.0 - G_2231[u,t,a,i] * 2.0
  G_2231 = 0

  # h_2222 --> G2222
  G_2222 = G_rtsu_calc(2,2)
  G_2222 .= G_2222 .* 2.0
  @tensoropt G_2222[t,u,v,w] -= Itu[u,w] * A[occ1o,occ1o][t,v]
  @tensoropt h_2222[t,u,v,w] := G_2222[t,u,v,w] - G_2222[u,t,v,w] - G_2222[t,u,w,v] + G_2222[u,t,w,v]
  G_2222 = 0

  # h_3221 --> G3221, G3221
  h_3221 = G_rtsj_calc(3,2)
  h_3221 .= h_3221 .* 2
  G_3212 = G_rtsu_calc(3,1)
  @tensoropt h_3221[a,t,u,i] -= G_3212[a,t,i,u] * 2.0 - Itu[t,u] * A[occv,occ2][a,i]
  G_3212 = 0

  # h_3231 --> G3231
  h_3231 = G_rtsj_calc(3,3)
  h_3231 .= 2.0 .* h_3231
  @tensoropt h_3231[a,t,b,i] -= Iab[a,b] * A[occ1o,occ2][t,i]

  # h_3222 --> G3222 twice
  G_3222 = G_rtsu_calc(3,2)
  G_3222 .= G_3222 .* 2.0
  @tensoropt G_3222[a,t,u,v] -= Itu[t,v] * A[occv,occ1o][a,u]
  @tensoropt h_3222[a,t,u,v] := G_3222[a,t,u,v] - G_3222[a,t,v,u]
  G_3222 = 0

  # h_3232 --> G3232
  h_3232 = G_rtsu_calc(3,3)
  h_3232 .= h_3232 .* 2.0
  @tensoropt h_3232[a,t,b,u] -= Iab[a,b] * A[occ1o,occ1o][t,u]

  h_2121 = reshape(h_2121, num_MO[2]*num_MO[1], num_MO[2]*num_MO[1])
  h_3121 = reshape(h_3121, num_MO[3]*num_MO[1], num_MO[2]*num_MO[1])
  h_3131 = reshape(h_3131, num_MO[3]*num_MO[1], num_MO[3]*num_MO[1])
  h_2221 = reshape(h_2221, num_MO[2]*num_MO[2], num_MO[2]*num_MO[1])
  h_2231 = reshape(h_2231, num_MO[2]*num_MO[2], num_MO[3]*num_MO[1])
  h_2222 = reshape(h_2222, num_MO[2]*num_MO[2], num_MO[2]*num_MO[2])
  h_3221 = reshape(h_3221, num_MO[3]*num_MO[2], num_MO[2]*num_MO[1])
  h_3231 = reshape(h_3231, num_MO[3]*num_MO[2], num_MO[3]*num_MO[1])
  h_3222 = reshape(h_3222, num_MO[3]*num_MO[2], num_MO[2]*num_MO[2])
  h_3232 = reshape(h_3232, num_MO[3]*num_MO[2], num_MO[3]*num_MO[2])
  return h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232
end

"""
    function calc_h_SCI(EC::ECInfo, fock::Matrix, D1::Matrix, D2, h_SO)

Calculate the hessian matrix with first order super-CI method
"""
function calc_h_SCI(EC::ECInfo, cMO::Matrix, fock::Matrix, D1::Matrix, D2, h_SO)
  n_MO = size(cMO,2)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:n_MO, EC.space['o']) # to be modified
  n_2 = size(occ2, 1)
  n_1o = size(occ1o, 1)
  n_v = size(occv, 1)
  I_ij = 1.0 * Matrix(I, n_2, n_2)
  I_tu = 1.0 * Matrix(I, n_1o, n_1o)
  I_ab = 1.0 * Matrix(I, n_v, n_v)
  fock = cMO' * fock * cMO
  @tensoropt begin
    h_ai_bj[a,i,b,j] := 4 * I_ij[i,j] * fock[occv,occv][a,b] - 4 * I_ab[a,b] * fock[occ2,occ2][i,j]
    h_ai_bu[a,i,b,u] := -2 * I_ab[a,b] * fock[occ2,occ1o][i,v] * D1[v,u]
    h_ai_uj[a,i,u,j] := I_ij[i,j] * (4 * fock[occv,occ1o][a,u] - 2 * fock[occv,occ1o][a,v]*D1[v,u]) 
    h_ti_uj[t,i,u,j] := (2*D1[t,u] - 4*I_tu[t,u]) * fock[occ2,occ2][i,j] +
                                  2 * I_ij[i,j] * (2*fock[occ1o,occ1o][t,u] - (D2[t,u,v,w] - D1[t,u]*D1[v,w]) * fock[occ1o,occ1o][v,w]-
                                      D1[t,v] * fock[occ1o,occ1o][v,u] - D1[v,u] * fock[occ1o,occ1o][t,v])
    h_at_bu[a,t,b,u] := 2*I_ab[a,b] * (D2[t,u,v,w] - D1[t,u] * D1[v,w]) * fock[occ1o,occ1o][v,w] +2*D1[t,u]*fock[occv,occv][a,b] 
  end
  h = zeros((n_MO,n_MO,n_MO,n_MO))
  #h = zeros((n_1o+n_v,n_2+n_1o,n_1o+n_v,n_2+n_1o))
  h[occv,occ2,occv,occ2] = h_ai_bj
  h[occv,occ2,occv,occ1o] = h_ai_bu
  h[occv,occ1o,occv,occ2] = permutedims(h_ai_bu, [3,4,1,2])
  h[occv,occ2,occ1o,occ2] = h_ai_uj
  h[occ1o,occ2,occv,occ2] = permutedims(h_ai_uj, [3,4,1,2])
  h[occ1o,occ2,occ1o,occ2] = h_ti_uj
  h[occv,occ1o,occv,occ1o] = h_at_bu
  h[occ1o,occ1o,:,:] = h_SO[occ1o,occ1o,:,:]
  h[:,:,occ1o,occ1o] = h_SO[:,:,occ1o,occ1o]
  h_rk_sl = h[[occ1o;occv],[occ2;occ1o],[occ1o;occv],[occ2;occ1o]]
  d = (n_1o+n_v)*(n_2+n_1o)
  h_rk_sl = reshape(h_rk_sl, d, d)
  return h_rk_sl
end

"""
    function calc_h_combined(EC::ECInfo, fock::Matrix, D1::Matrix, D2, h_SO)

Calculate the hessian matrix with combination of Super CI and Second Order Approximation methods
"""
function calc_h_combined(EC::ECInfo, cMO::Matrix, fock::Matrix, D1::Matrix, D2, h_SO)
  n_MO = size(cMO,2)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:n_MO, EC.space['o']) # to be modified
  n_2 = size(occ2, 1)
  n_1o = size(occ1o, 1)
  n_v = size(occv, 1)
  I_ij = 1.0 * Matrix(I, n_2, n_2)
  I_ab = 1.0 * Matrix(I, n_v, n_v)
  I_tu = 1.0 * Matrix(I, n_1o, n_1o)
  fock = cMO' * fock * cMO
  @tensoropt begin
    h_ai_bj[a,i,b,j] := 4 * I_ij[i,j] * fock[occv,occv][a,b] - 4 * I_ab[a,b] * fock[occ2,occ2][i,j]
    h_ai_bu[a,i,b,u] := -2 * I_ab[a,b] * fock[occ2,occ1o][i,v] * D1[v,u]
    h_ai_uj[a,i,u,j] := I_ij[i,j] * (4 * fock[occv,occ1o][a,u] - 2 * fock[occv,occ1o][a,v]*D1[v,u])
  end
  h = zeros((n_MO,n_MO,n_MO,n_MO))
  #h = zeros((n_1o+n_v,n_2+n_1o,n_1o+n_v,n_2+n_1o))
  h[occv,occ2,occv,occ2] = h_ai_bj
  h[occv,occ2,occv,occ1o] = h_ai_bu
  h[occv,occ1o,occv,occ2] = permutedims(h_ai_bu, [3,4,1,2])
  h[occv,occ2,occ1o,occ2] = h_ai_uj
  h[occ1o,occ2,occv,occ2] = permutedims(h_ai_uj, [3,4,1,2])
  h[occ1o,occ2,occ1o,occ2] = h_SO[occ1o,occ2,occ1o,occ2]
  h[occv,occ1o,occv,occ1o] = h_SO[occv,occ1o,occv,occ1o] 
  h[occ1o,occ1o,:,:] = h_SO[occ1o,occ1o,:,:]
  h[:,:,occ1o,occ1o] = h_SO[:,:,occ1o,occ1o]
  h_rk_sl = h[[occ1o;occv],[occ2;occ1o],[occ1o;occv],[occ2;occ1o]]
  d = (n_1o+n_v)*(n_2+n_1o)
  h_rk_sl = reshape(h_rk_sl, d, d)
  return h_rk_sl
end


"""
    calc_realE(EC::ECInfo, fockClosed::Matrix, D1::Matrix, D2, cMO::Matrix)

Calculate the energy with the given density matrices and (updated) cMO, 
``E = (h_i^i + ^cf_i^i) + ^1D^t_u ^cf_t^u + 0.5 ^2D^{tv}_{uw} v_{tv}^{uw}``.
"""
function calc_realE(EC::ECInfo, fockClosed::Matrix, D1::Matrix, D2, cMO::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  hsmall = load(EC,"hsmall")
  CMO2 = cMO[:,occ2] 
  CMOa = cMO[:,occ1o] 
  #μνL = load(EC,"munuL")
  μuL = load(EC,"muaL")
  @tensoropt E = CMO2[μ,i]*(hsmall[μ,ν]+fockClosed[μ,ν])*CMO2[ν,i]
  @tensoropt fockClosed_MO[t,u] := fockClosed[μ,ν] * CMOa[μ,t] *CMOa[ν,u]
  E += sum(fockClosed_MO .* D1)
  @tensoropt tuL[t,u,L] := μuL[μ,u,L] * CMOa[μ,t]
  @tensoropt tuvw[t,u,v,w] := tuL[t,u,L] * tuL[v,w,L]
  E += 0.5 * sum(D2 .* tuvw)
  return E
end

"""
    davidson(H::Matrix, N::Integer, n::Integer, thres::Number, convTrack::Bool=false)

Calculate one of the eigenvalues and corresponding eigenvector of the matrix H
(usually the lowest eigenvalue), 
N is the size of the matrix H, 
n_max is the maximal size of projected matrix, 
thres is the criterion of convergence, 
convTrack is to decide whether the tracking of eigenvectors is used
"""
function davidson(v::Vector, N::Integer, n_max::Integer, thres::Number,  num_MO::Vector{Int64},
  h_block::NTuple{10, Matrix{Float64}}, g::Vector, α::Number, initVecType::InitialVectorType, convTrack::Bool=false)
  V = zeros(N,n_max)
  σ = zeros(N,n_max)
  h = zeros(n_max,n_max)
  ac = zeros(n_max)
  λ = zeros(n_max)
  eigvec_index = 1
  pick_vec = 6
  converged = false
  n_2, n_1o, n_v = num_MO
  n21 = n_1o * n_2
  n31 = n_v * n_2
  n22 = n_1o * n_1o
  n32 = n_v * n_1o
  h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232 = h_block
  H0_hb = [[0.];diag(h_2121);diag(h_3131);diag(h_2222);diag(h_3232)]
  initGuessIndex = findmax(abs.(H0_hb))[2]
  numInitialVectors = 0

  function H_multiply(v::Vector)
    v21 = v[2:n21+1] 
    v31 = v[n21+2:n21+n31+1]
    v22 = v[n21+n31+2:n21+n31+n22+1]
    v32 = v[n21+n31+n22+2:end]
    @tensoropt newσ_1[n] := h_2121[n,m] * v21[m]
    @tensoropt newσ_1[n] += h_3121[m,n] * v31[m]
    @tensoropt newσ_1[n] += h_2221[m,n] * v22[m]
    @tensoropt newσ_1[n] += h_3221[m,n] * v32[m]
    @tensoropt newσ_2[n] := h_3121[n,m] * v21[m]
    @tensoropt newσ_2[n] += h_3131[n,m] * v31[m]
    @tensoropt newσ_2[n] += h_2231[m,n] * v22[m]
    @tensoropt newσ_2[n] += h_3231[m,n] * v32[m]
    @tensoropt newσ_3[n] := h_2221[n,m] * v21[m]
    @tensoropt newσ_3[n] += h_2231[n,m] * v31[m]
    @tensoropt newσ_3[n] += h_2222[n,m] * v22[m]
    @tensoropt newσ_3[n] += h_3222[m,n] * v32[m]
    @tensoropt newσ_4[n] := h_3221[n,m] * v21[m]
    @tensoropt newσ_4[n] += h_3231[n,m] * v31[m]
    @tensoropt newσ_4[n] += h_3222[n,m] * v22[m]
    @tensoropt newσ_4[n] += h_3232[n,m] * v32[m]
    newσ_hb = [[g'*v[2:end].* α];newσ_1;newσ_2;newσ_3;newσ_4]
    newσ_hb[2:end] .+= g .* v[1] .*α
    return newσ_hb
  end

  if initVecType == RANDOM
    v = rand(size(v,1))
    v = v ./ norm(v)
    numInitialVectors = 1
    V[:,1] = v
  elseif initVecType == INHERIT
    v = v ./ norm(v)
    numInitialVectors = 1
    V[:,1] = v
  elseif initVecType == GRADIENT_SET
    V[1,1] = 1.0
    g_r = g + rand(size(g,1)) .* 0.02 .- 0.01
    v = [[0.];g_r] ./ norm(g_r)
    V[:,2] = v
    σ[:,1] = H_multiply(V[:,1])
    numInitialVectors = 2
  elseif initVecType == GRADIENT_SETPLUS
    V[1,1] = 1.0
    g_r = g + rand(size(g,1)) .* 0.02 .- 0.01
    v = [[0.];g_r] ./ norm(g_r)
    V[:,2] = v
    σ[:,1] = H_multiply(V[:,1])
    σ[:,2] = H_multiply(V[:,2])
    V[initGuessIndex, 3] = 1.0
    v = V[:,3]
    v = v - V[initGuessIndex,2].* V[:,2]
    v = v ./ norm(v)
    V[:,3] = v
    newh_hb = V' * σ[:,2]
    h[:,2] = newh_hb
    h[2,:] = newh_hb
    numInitialVectors = 3
  end
  
  for i in numInitialVectors+1:n_max
    # blockwise H * v
    newσ_hb = H_multiply(v)
    σ[:,i-1] = newσ_hb
    newh_hb = V' * newσ_hb
    h[:,i-1] = newh_hb 
    h[i-1,:] = newh_hb
    λ, a = eigen(Hermitian(h[1:i-1,1:i-1]))
    if convTrack && i > pick_vec
      eigvec_index = findmax(abs.(ac[1:i-1]' * a[:,1:pick_vec]))[2][2]
    end
    ac[1:i-1] = a[:,eigvec_index]
    r = σ * ac - λ[eigvec_index] * (V * ac)
    if norm(r) < thres
      converged = true
      println("Davidson iter ", i, " converged!")
      break
    end
    v = -1.0 ./ (H0_hb .- λ[eigvec_index]) .* r
    c = transpose(v) * V
    v = v - V * transpose(c)
    v = v./norm(v)
    V[:,i] = v
  end
  if !converged
    println("davidson algorithm not converged!")
  end
  v = V * ac
  return λ[eigvec_index], v, converged
end

"""
    λTuning(trust::Number, maxit::Integer, λmax::Number, λ::Number, h::Matrix, g::Vector)

Find the rotation parameters as the vector x in trust region,
tuning λ with the norm of x in the iterations.
Return λ and x.
"""
function λTuning(trust::Number, maxit::Integer, αmax::Number, α::Number, g::Vector, vec::Vector, num_MO::Vector{Int64}, 
  h_block::NTuple{10, Matrix{Float64}}, initVecType::InitialVectorType)
  N_rk = (num_MO[2]+num_MO[3]) * (num_MO[1]+num_MO[2])
  g_norm = norm(g)
  x = zeros(N_rk)
  αl = 1.0
  αr = αmax
  micro_converged = false
  davItMax = 200 # for davidson eigenvalue solving algorithm
  davError = 1e-7
  γ =  0.1 # gradient scaling factor for micro-iteration accuracy
  davError = γ * norm(g)
  # α tuning loop (micro loop)
  for it=1:maxit
    @timeit "davidson" val, vec, converged = davidson(vec, N_rk+1, davItMax, davError, num_MO, h_block, g, α, initVecType)
    micro_counts = 0
    while !converged
      davItMax += 50
      println("Davidson max iteration number increased to ", davItMax)
      @timeit "davidson" val, vec, converged = davidson(vec, N_rk+1, davItMax, davError, num_MO, h_block, g, α, initVecType)
      micro_counts += 1
    end
    x = vec[2:end] ./ (vec[1] * α)
    # check if square of norm of x in trust region (0.8*trust ~ trust)
    sumx2 = sum(x.^2)
    if sumx2 > trust
      αl = α
    elseif sumx2 < 0.8*trust
      αr = α
    else
      micro_converged = true
      break
    end
    if αr ≈ αl || g_norm<1e-3
      α = αl
      micro_converged = true
      break
    end
    α = (αl + αr) / 2.0
  end
  if !micro_converged
    println("micro NOT converged")
  end
  return α, x, vec
end

"""
    calc_U(EC::ECInfo, N_MO::Integer, x::Vector)

calculate orbital-rotational matrix U (approximately unitary because of the anti-hermitian property of the R
which is constructed from `x`).
"""
function calc_U(EC::ECInfo, N_MO::Integer, x::Vector)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:N_MO, EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  R = zeros(N_MO,N_MO)
  R[occ1o,occ2] = reshape(x[1:n_1o*n_2], n_1o, n_2)
  R[occv,occ2] = reshape(x[n_1o*n_2+1:n_1o*n_2+n_v*n_2], n_v, n_2)
  R[occv,occ1o] = reshape(x[n_1o*n_2+n_v*n_2+n_1o*n_1o+1:end], n_v, n_1o)
  R = R-R'
  R[occ1o,occ1o] = reshape(x[n_1o*n_2+n_v*n_2+1:n_1o*n_2+n_v*n_2+n_1o*n_1o], n_1o, n_1o)
  U = 1.0 * Matrix(I,N_MO,N_MO) + R
  U = U + 1/2 .* R*R + 1/6 .* R*R*R + 1/24 .* R*R*R*R
  return U
end

"""
    checkE_modifyTrust(E::Number, E_former::Number, E_2o::Number, trust::Number)

Check if the energy E is lower than the former energy E_former,
if not, reject the update of coefficients and modify the trust region.
Return reject::Bool and trust.
"""
function checkE_modifyTrust(E, E_former, E_2o, trust)
  energy_diff = E - E_former
  energy_quotient = energy_diff / E_2o
  # modify the trust region
  reject = false
  if energy_quotient < 0.0 || E_2o > 0.0
    trust = 0.7 * trust
    reject = true
  elseif energy_quotient < 0.25
    trust = 0.7 * trust
  elseif energy_quotient > 0.75
    trust = 1.2 * trust
  end
  return reject, trust
end

"""
    dfmcscf(EC::ECInfo; direct = false, guess = GUESS_SAD, IterMax=50)

Main body of Density-Fitted Multi-Configurational Self-Consistent-Field method
"""
function dfmcscf(EC::ECInfo; direct = false, guess = GUESS_SAD, IterMax=64, maxit=16)
  initVecType::InitialVectorType = GRADIENT_SETPLUS
  Enuc = generate_integrals(EC; save3idx=!direct)
  println("Enuc ", Enuc)
  sao = load(EC,"sao")
  nAO = size(sao,2) # number of atomic orbitals
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:nAO, EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  num_MO = [n_2,n_1o,n_v]
  N_rk = (n_1o+n_v) * (n_2+n_1o)
  vec = rand(N_rk+1)
  vec = vec ./ norm(vec)
  inherit_large = true
  reject = false
  if size(occ1o,1) == 0
    error("NO ACTIVE ORBITALS, PLEASE USE DFHF")
  end

  # cMO and density matrices initialization
  cMO = guess_orb(EC,guess)
  D1, D2 = denMatCreate(EC)

  # macro loop parameters
  iteration_times = 0
  g = [1]
  E_former = 0.0
  trust = 0.4
  λ = 500.0

  fock = zeros(nAO,nAO)
  fockClosed = zeros(nAO,nAO)
  prev_cMO = deepcopy(cMO)
  E_2o = 0.0
  E = 0.0

  @timeit "iterations" begin
    # macro loop, g and h updated
    while iteration_times < IterMax

      # calc energy E with updated cMO
      @timeit "fock calc" fock, fockClosed = dffockCAS(EC,cMO,D1)
      @timeit "E calc" E = calc_realE(EC, fockClosed, D1, D2, cMO)
      # check if reject the update and tune trust
      if iteration_times > 0
        reject, trust = checkE_modifyTrust(E, E_former, E_2o, trust)
        if reject
          cMO = prev_cMO
          @timeit "fock calc" fock, fockClosed = dffockCAS(EC,cMO,D1)
          E = E_former
          iteration_times -= 1
          inherit_large = false
        end
      end

      println("Iter ", iteration_times, " energy: ", E+Enuc)

      iteration_times += 1
      E_former = E
      # calc g and h with updated cMO
      @timeit "A calc" A = dfACAS(EC,cMO,D1,D2,fock,fockClosed)
      @timeit "g calc" g = calc_g(A, EC)
      n21 = n_1o * n_2
      n31 = n_v * n_2
      n22 = n_1o * n_1o
      n32 = n_v * n_1o
      println("norm of g: ", norm(g))
      if norm(g) < 1e-5 
        break
      end
      @timeit "h calc new" h_block = calc_h_SO(EC, cMO, D1, D2, fock, fockClosed, A)
      # h = calc_h_SCI(EC, cMO, fock, D1, D2, h_SO)
      # h = calc_h_combined(EC, cMO, fock, D1, D2, h_SO)

      # λ tuning loop (micro loop)
      λmax = 1000.0
      if inherit_large == false
        vec = rand(N_rk+1)
        vec = vec./norm(vec)
        inherit_large == true
      end
      if norm(g) > 1e-3
        @timeit "λTuning" λ, x, vec = λTuning(trust, maxit, λmax, λ, g, vec, num_MO, h_block, initVecType)
      end
      # calc 2nd order perturbation energy
      h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232 = h_block

      H_matrix = [[h_2121;h_3121;h_2221;h_3221] [h_3121';h_3131;h_2231;h_3231] [h_2221';h_2231';h_2222;h_3222] [h_3221';h_3231';h_3222';h_3232]]
      if norm(g) < 1e-3
        # I_rk = 1.0 * Matrix(I, N_rk-n22, N_rk-n22)
        I_rk = 1.0 * Matrix(I, N_rk, N_rk)
        H_matrix += rand() * 1e-10 .* I_rk
        # x_no22 = - H_matrix \ [g[1:n21+n31];g[end-n32+1:end]]
        # x = [x_no22[1:n21+n31];zeros(n22);x_no22[end-n32+1:end]]
        x = - H_matrix \ g
        println("square of the norm of x: ", sum(x.^2))      
      end

      function calc_E_2o(x)
        x21 = x[1:n21] 
        x31 = x[n21+1:n21+n31]
        x22 = x[n21+n31+1:n21+n31+n22]
        x32 = x[n21+n31+n22+1:end]
        @tensoropt newσ_1[n] := h_2121[n,m] * x21[m]
        @tensoropt newσ_1[n] += h_3121[m,n] * x31[m]
        @tensoropt newσ_1[n] += h_2221[m,n] * x22[m]
        @tensoropt newσ_1[n] += h_3221[m,n] * x32[m]
        @tensoropt newσ_2[n] := h_3121[n,m] * x21[m]
        @tensoropt newσ_2[n] += h_3131[n,m] * x31[m]
        @tensoropt newσ_2[n] += h_2231[m,n] * x22[m]
        @tensoropt newσ_2[n] += h_3231[m,n] * x32[m]
        @tensoropt newσ_3[n] := h_2221[n,m] * x21[m]
        @tensoropt newσ_3[n] += h_2231[n,m] * x31[m]
        @tensoropt newσ_3[n] += h_2222[n,m] * x22[m]
        @tensoropt newσ_3[n] += h_3222[m,n] * x32[m]
        @tensoropt newσ_4[n] := h_3221[n,m] * x21[m]
        @tensoropt newσ_4[n] += h_3231[n,m] * x31[m]
        @tensoropt newσ_4[n] += h_3222[n,m] * x22[m]
        @tensoropt newσ_4[n] += h_3232[n,m] * x32[m]
        E_2o = sum(g .* x) + 0.5*(transpose(x) * [newσ_1;newσ_2;newσ_3;newσ_4])
        return E_2o
      end
      # scale = 1.5
      # x = x .* 2.0
      E_2o_c = calc_E_2o(x)
      # increase = true
      # while(increase && scale < 10.0)
      #   E_2o_trial = calc_E_2o(x.*scale)
      #   if E_2o_trial > E_2o_c
      #     x = x .* scale ./ 1.5
      #     increase = false
      #   else
      #     println("scale increased to ", scale)
      #     scale = scale * 1.5
      #     E_2o_c = E_2o_trial
      #   end
      # end
      E_2o = E_2o_c
      
      # calc rotation matrix U
      U = calc_U(EC, nAO, x)
      #println("difference between U and a real unitary matrix: ", sum((U'*U-I).^2))

      # update cMO with U
      prev_cMO = deepcopy(cMO)
      cMO = cMO*U

      # reorthogonalize molecular orbitals
      smo = cMO' * sao * cMO
      cMO = cMO * Hermitian(smo)^(-1/2)
    end
  end

  if iteration_times < IterMax
    println("Convergent!")
  else
    println("Not Convergent!")
  end
  return E_former+Enuc, cMO
end
end #module
