module DFMCSCF
using LinearAlgebra, TensorOperations, Printf, TimerOutputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.DIIS
using ..ElemCo.TensorTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.DFHF

export dfmcscf

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
function dffockCAS(μνL, μjL, μuL, EC::ECInfo, cMO::Matrix, D1::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified

  # fockClosed
  hsmall = load(EC,"h_AA")
  @tensoropt L[L] := μjL[μ,j,L] * CMO2[μ,j]
  @tensoropt fockClosed[μ,ν] := hsmall[μ,ν] - μjL[μ,j,L]*μjL[ν,j,L]
  @tensoropt fockClosed[μ,ν] += 2.0*L[L]*μνL[μ,ν,L]

  # fock
  fock =  deepcopy(fockClosed)
  @tensoropt μuLD[μ,t,L] := μuL[μ,u,L] * D1[t,u]
  @tensoropt fock[μ,ν] -= 0.5 * μuLD[μ,t,L] * μuL[ν,t,L]
  @tensoropt LD[L] := μuLD[μ,t,L] * CMOa[μ,t]
  @tensoropt fock[μ,ν] += LD[L] * μνL[μ,ν,L]

  @tensoropt fock_MO[r,s] := fock[μ,ν] * cMO[μ,r] * cMO[ν,s]
  @tensoropt fockClosed_MO[r,s] := fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s]
  return fock_MO, fockClosed_MO, μjL, μuL
end

"""
    dfACAS(EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock::Matrix, fockClosed::Matrix)

Calculate the A-intermediate matrix in molecular orbital basis.
return matrix A[p,q]
"""
function dfACAS(μuL, EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock_MO::Matrix, fockClosed_MO::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  # Apj
  Apj = fock_MO[:, occ2] .* 2.0
  # Apu
  @tensoropt Apu[p,u] := fockClosed_MO[:,occ1o][p,v] * D1[v,u]
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

function G_risj_calc(typer::Integer, types::Integer, num_MO, index_MO, fock_MO::Matrix, DFint_MO)
  n_2,n_1o,n_v = num_MO
  Iij = 1.0 * Matrix(I,n_2,n_2)
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
  return G_risj
end

function G_rtsj_calc(typer::Integer,types::Integer, num_MO, DFint_MO, D1::Matrix)
  n_2,n_1o,n_v = num_MO
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

function G_rtsu_calc(typer::Integer,types::Integer, num_MO, index_MO, DFint_MO, D1::Matrix, D2, fockClosed_MO::Matrix)
  n_2,n_1o,n_v = num_MO
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

function h_calc_fixed(num_MO, index_MO, D1::Matrix, D2, fockClosed_MO::Matrix, A::Matrix, DFint_MO)
  n_2,n_1o,n_v = num_MO
  occ2,occ1o,occv = index_MO
  Itu = 1.0 * Matrix(I,n_1o,n_1o)

  # h_2221 --> G2221, G2212 each for twice
  G_2221 = G_rtsj_calc(2,2, num_MO, DFint_MO, D1)
  @tensoropt h_2221[t,u,v,i] := G_2221[t,u,v,i] * 2.0 - G_2221[u,t,v,i] * 2.0
  G_2212 = G_rtsu_calc(2,1, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  @tensoropt h_2221[t,u,v,i] += -2.0 * G_2212[t,u,i,v] + 2.0 * G_2212[u,t,i,v]
  @tensoropt h_2221[t,u,v,i] += Itu[u,v] * A[occ1o,occ2][t,i]
  @tensoropt h_2221[t,u,v,i] -= Itu[t,v] * A[occ1o,occ2][u,i]
  G_2221 = 0
  G_2212 = 0

  # h_2231 --> G2231 twice
  G_2231 = G_rtsj_calc(2,3, num_MO, DFint_MO, D1)
  @tensoropt h_2231[t,u,a,i] := G_2231[t,u,a,i] * 2.0 - G_2231[u,t,a,i] * 2.0
  G_2231 = 0

  # h_2222 --> G2222
  G_2222 = G_rtsu_calc(2,2, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  G_2222 .= G_2222 .* 2.0
  @tensoropt G_2222[t,u,v,w] -= Itu[u,w] * A[occ1o,occ1o][t,v]
  @tensoropt h_2222[t,u,v,w] := G_2222[t,u,v,w] - G_2222[u,t,v,w] - G_2222[t,u,w,v] + G_2222[u,t,w,v]
  G_2222 = 0

  # h_3222 --> G3222 twice
  G_3222 = G_rtsu_calc(3,2, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  G_3222 .= G_3222 .* 2.0
  @tensoropt G_3222[a,t,u,v] -= Itu[t,v] * A[occv,occ1o][a,u]
  @tensoropt h_3222[a,t,u,v] := G_3222[a,t,u,v] - G_3222[a,t,v,u]
  G_3222 = 0 
  return h_2221, h_2231, h_2222, h_3222
end

function h_calc_SOpart(num_MO, index_MO, DFint_MO, D1::Matrix, D2, fock_MO::Matrix, fockClosed_MO::Matrix, A::Matrix)
  n_2,n_1o,n_v = num_MO
  occ2,occ1o,occv = index_MO
  Iij = 1.0 * Matrix(I,n_2,n_2)
  Itu = 1.0 * Matrix(I,n_1o,n_1o)
  h_3221 = G_rtsj_calc(3,2, num_MO, DFint_MO, D1)
  h_3221 .= h_3221 .* 2
  G_3212 = G_rtsu_calc(3,1, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  @tensoropt h_3221[a,t,u,i] -= G_3212[a,t,i,u] * 2.0 - Itu[t,u] * A[occv,occ2][a,i]
  G_3212 = 0

  h_2121 = G_risj_calc(2,2, num_MO, index_MO, fock_MO, DFint_MO)
  h_2121 .= h_2121 .* 2.0
  @tensoropt h_2121[t,i,u,j] -= Iij[i,j] * A[occ1o,occ1o][t,u]
  G_1221 = G_rtsj_calc(1,2, num_MO, DFint_MO, D1)
  @tensoropt h_2121[t,i,u,j] -= G_1221[i,t,u,j] * 2.0
  @tensoropt h_2121[t,i,u,j] -= G_1221[j,u,t,i] * 2.0
  G_1212 = G_rtsu_calc(1,1, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  @tensoropt h_2121[t,i,u,j] += G_1212[i,t,j,u] * 2.0
  @tensoropt h_2121[t,i,u,j] -= Itu[t,u] * A[occ2,occ2][i,j]
  G_1221 = 0
  G_1212 = 0

  return h_3221, h_2121
end

function calc_h_SO(μjL, μuL, abL, EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock_MO::Matrix, fockClosed_MO::Matrix, A::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  num_MO = [n_2,n_1o,n_v]
  index_MO = [occ2,occ1o,occv]
  A = A + A'

  # precalculate the density fitting integrals in molecular orbital basis
  @tensoropt ijL[i,j,L] := μjL[μ,j,L] * cMO[:,occ2][μ,i]
  @tensoropt tiL[t,i,L] := μjL[μ,i,L] * cMO[:,occ1o][μ,t]
  @tensoropt aiL[a,i,L] := μjL[μ,i,L] * cMO[:,occv][μ,a]
  @tensoropt tuL[t,u,L] := μuL[μ,u,L] * cMO[:,occ1o][μ,t]
  @tensoropt atL[a,t,L] := μuL[μ,t,L] * cMO[:,occv][μ,a]

  DFint_MO = [[ijL,tiL,aiL],[tiL,tuL,atL],[aiL,atL,abL]]
  Iij = 1.0 * Matrix(I,n_2,n_2)
  Iab = 1.0 * Matrix(I,n_v,n_v)

  h_3221, h_2121 = h_calc_SOpart(num_MO, index_MO, DFint_MO, D1, D2, fock_MO, fockClosed_MO, A)

  # h_3131 ==> G3131 needed, the largest and most memory consuming part
  h_3131 = G_risj_calc(3,3, num_MO, index_MO, fock_MO, DFint_MO)
  h_3131 .= h_3131 .* 2.0
  @tensoropt h_3131[a,i,b,j] -= Iab[a,b] * A[occ2,occ2][i,j]

  # h_3121 --> G3121, G1231 needed
  h_3121 = G_risj_calc(3,2, num_MO, index_MO, fock_MO, DFint_MO)
  h_3121 .= h_3121 .* 2.0
  @tensoropt h_3121[a,i,t,j] -= Iij[i,j] * A[occv,occ1o][a,t]
  G_1231 = G_rtsj_calc(1,3, num_MO, DFint_MO, D1)
  @tensoropt h_3121[a,i,t,j] -= G_1231[j,t,a,i] * 2.0

  # h_3231 --> G3231
  h_3231 = G_rtsj_calc(3,3, num_MO, DFint_MO, D1)
  h_3231 .= 2.0 .* h_3231
  @tensoropt h_3231[a,t,b,i] -= Iab[a,b] * A[occ1o,occ2][t,i]

  # h_3232 --> G3232
  h_3232 = G_rtsu_calc(3,3, num_MO, index_MO, DFint_MO, D1, D2, fockClosed_MO)
  h_3232 .= h_3232 .* 2.0
  @tensoropt h_3232[a,t,b,u] -= Iab[a,b] * A[occ1o,occ1o][t,u]

  h_2221, h_2231, h_2222, h_3222 = h_calc_fixed(num_MO, index_MO, D1, D2, fockClosed_MO, A, DFint_MO)

  h_2121 = reshape(h_2121, num_MO[2]*num_MO[1], num_MO[2]*num_MO[1])
  h_3121 = reshape(h_3121, num_MO[3]*num_MO[1], num_MO[2]*num_MO[1])
  h_3131 = reshape(h_3131, num_MO[3]*num_MO[1], num_MO[3]*num_MO[1])
  h_3231 = reshape(h_3231, num_MO[3]*num_MO[2], num_MO[3]*num_MO[1])
  h_2221 = reshape(h_2221, num_MO[2]*num_MO[2], num_MO[2]*num_MO[1])
  h_2231 = reshape(h_2231, num_MO[2]*num_MO[2], num_MO[3]*num_MO[1])
  h_2222 = reshape(h_2222, num_MO[2]*num_MO[2], num_MO[2]*num_MO[2])
  h_3221 = reshape(h_3221, num_MO[3]*num_MO[2], num_MO[2]*num_MO[1])
  h_3222 = reshape(h_3222, num_MO[3]*num_MO[2], num_MO[2]*num_MO[2])
  h_3232 = reshape(h_3232, num_MO[3]*num_MO[2], num_MO[3]*num_MO[2])
  return h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232
end

function calc_h_SCI(μjL, μuL, EC::ECInfo, cMO::Matrix, D1::Matrix, D2, fock_MO::Matrix, fockClosed_MO::Matrix, A::Matrix, HessianType::Symbol=:SCI)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:size(cMO,2), EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  num_MO = [n_2,n_1o,n_v]
  index_MO = [occ2,occ1o,occv]
  abL = zeros(1,1)
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

  h_3131 = zeros(1,1)
  h_3121 = zeros(1,1)
  h_3231 = zeros(1,1)

  if HessianType == :SCI
    h_3221 = zeros(n_v,n_1o,n_1o,n_2)
    @tensoropt h_2121[t,i,u,j] := (2*D1[t,u] - 4*Itu[t,u]) * fock_MO[occ2,occ2][i,j] 
    @tensoropt h_2121[t,i,u,j] += 2 * Iij[i,j] * (2*fock_MO[occ1o,occ1o][t,u] 
      -(D2[t,u,v,w] - D1[t,u]*D1[v,w]) * fock_MO[occ1o,occ1o][v,w]
      -D1[t,v] * fock_MO[occ1o,occ1o][v,u] - D1[v,u] * fock_MO[occ1o,occ1o][t,v])    
  elseif HessianType == :SO_SCI
    h_3221, h_2121 = h_calc_SOpart(num_MO, index_MO, DFint_MO, D1, D2, fock_MO, fockClosed_MO, A)
  end

  # h_3232 --> G3232
  @tensoropt h_3232[a,t,b,u] := 2.0 * Iab[a,b] * (D2[t,u,v,w] - D1[t,u] * D1[v,w]) * fock_MO[occ1o,occ1o][v,w] 
  @tensoropt h_3232[a,t,b,u] += 2.0 * D1[t,u] * fock_MO[occv,occv][a,b] 

  h_2221, h_2231, h_2222, h_3222 = h_calc_fixed(num_MO, index_MO, D1, D2, fockClosed_MO, A, DFint_MO)

  h_2121 = reshape(h_2121, num_MO[2]*num_MO[1], num_MO[2]*num_MO[1])
  h_2221 = reshape(h_2221, num_MO[2]*num_MO[2], num_MO[2]*num_MO[1])
  h_2231 = reshape(h_2231, num_MO[2]*num_MO[2], num_MO[3]*num_MO[1])
  h_2222 = reshape(h_2222, num_MO[2]*num_MO[2], num_MO[2]*num_MO[2])
  h_3221 = reshape(h_3221, num_MO[3]*num_MO[2], num_MO[2]*num_MO[1])
  h_3222 = reshape(h_3222, num_MO[3]*num_MO[2], num_MO[2]*num_MO[2])
  h_3232 = reshape(h_3232, num_MO[3]*num_MO[2], num_MO[3]*num_MO[2])
  return h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232
end

"""
    calc_realE(EC::ECInfo, fockClosed::Matrix, D1::Matrix, D2, cMO::Matrix)

Calculate the energy with the given density matrices and (updated) cMO, 
``E = (h_i^i + ^cf_i^i) + ^1D^t_u ^cf_t^u + 0.5 ^2D^{tv}_{uw} v_{tv}^{uw}``.
"""
function calc_realE(μuL, EC::ECInfo, fockClosed_MO::Matrix, D1::Matrix, D2, cMO::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  hsmall = load(EC,"h_AA")
  CMO2 = cMO[:,occ2] 
  CMOa = cMO[:,occ1o]
  @tensoropt E = CMO2[μ,i] * hsmall[μ,ν] * CMO2[ν,i]
  E += tr(fockClosed_MO[occ2,occ2])
  E += sum(fockClosed_MO[occ1o, occ1o] .* D1)
  @tensoropt tuL[t,u,L] := μuL[μ,u,L] * CMOa[μ,t]
  @tensoropt tuvw[t,u,v,w] := tuL[t,u,L] * tuL[v,w,L]
  E += 0.5 * sum(D2 .* tuvw)
  return E
end

function Hx_common(h_2121, h_2221, h_3221, h_2231, h_2222, h_3222, h_3232, x)
  n21 = size(h_2121,1)
  n22, n31 = size(h_2231)
  x21 = x[1:n21] 
  x31 = x[n21+1:n21+n31]
  x22 = x[n21+n31+1:n21+n31+n22]
  x32 = x[n21+n31+n22+1:end]
  @tensoropt σ21[n] := h_2121[n,m] * x21[m]
  @tensoropt σ21[n] += h_2221[m,n] * x22[m]
  @tensoropt σ21[n] += h_3221[m,n] * x32[m]
  @tensoropt σ31[n] := h_2231[m,n] * x22[m]
  @tensoropt σ22[n] := h_2221[n,m] * x21[m]
  @tensoropt σ22[n] += h_2231[n,m] * x31[m]
  @tensoropt σ22[n] += h_2222[n,m] * x22[m]
  @tensoropt σ22[n] += h_3222[m,n] * x32[m]
  @tensoropt σ32[n] := h_3221[n,m] * x21[m]
  @tensoropt σ32[n] += h_3222[n,m] * x22[m]
  @tensoropt σ32[n] += h_3232[n,m] * x32[m]
  return [σ21;σ31;σ22;σ32]
end

function Hx_SO(h_3131, h_3231, h_3121, x, num_MO)
  n21 = size(h_3121,2)
  n31 = size(h_3231,2)
  n_2,n_1o,n_v = num_MO
  n22 = n_1o * n_1o
  x21 = x[1:n21] 
  x31 = x[n21+1:n21+n31]
  x32 = x[n21+n31+n22+1:end]
  @tensoropt σ21[n] := h_3121[m,n] * x31[m]
  @tensoropt σ31[n] := h_3131[n,m] * x31[m]
  @tensoropt σ31[n] += h_3231[m,n] * x32[m]
  @tensoropt σ31[n] += h_3121[n,m] * x21[m]
  σ22 = zeros(n22)
  @tensoropt σ32[n] := h_3231[n,m] * x31[m]
  return [σ21;σ31;σ22;σ32]
end

function Hx_SCI(EC::ECInfo, fock_MO::Matrix, cMO::Matrix, x::Vector, num_MO, D1::Matrix)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:size(cMO,1), EC.space['o']) # to be modified
  Fab = fock_MO[occv,occv]
  Fij = fock_MO[occ2,occ2]
  Fiv = fock_MO[occ2,occ1o]
  Fav = fock_MO[occv,occ1o]
  Fau = fock_MO[occv,occ1o]
  n_2,n_1o,n_v = num_MO
  n21 = n_1o * n_2
  n31 = n_v * n_2
  n22 = n_1o * n_1o
  x21 = x[1:n21] 
  x31 = x[n21+1:n21+n31]
  x32 = x[n21+n31+n22+1:end]
  x31_r = reshape(x31, n_v, n_2)
  x32_r = reshape(x32, n_v, n_1o)
  x21_r = reshape(x21, n_1o, n_2)
  @tensoropt σ21[u,i] := (-2.0 * Fav[a,v] * D1[v,u] + 4.0 * Fau[a,u])* x31_r[a,i]
  σ21 = reshape(σ21, n_2*n_1o)
  @tensoropt σ31[a,i] := 4.0 * Fab[a,b] * x31_r[b,i] - 4.0 * Fij[i,j] * x31_r[a,j]
  @tensoropt σ31[a,i] += -2.0 * Fiv[i,v] * x32_r[a,u] * D1[v,u]
  @tensoropt σ31[a,i] += (-2.0 * Fav[a,v] * D1[v,u] + 4.0 * Fau[a,u])* x21_r[u,i]
  σ31 = reshape(σ31, n_2*n_v)
  @tensoropt σ32[b,u] := -2.0 * Fiv[i,v] * x31_r[b,i] * D1[v,u]
  σ22 = zeros(n22)
  σ32 = reshape(σ32, n_1o*n_v)
  return [σ21;σ31;σ22;σ32]
end

function H_multiply(EC::ECInfo, fock_MO::Matrix, cMO::Matrix, D1::Matrix, v::Vector, num_MO, g::Vector, alpha::Number, 
  h_block::NTuple{10, Matrix{Float64}}, HessianType::Symbol)
  h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232 = h_block
  σ = Hx_common(h_2121, h_2221, h_3221, h_2231, h_2222, h_3222, h_3232, v[2:end])
  if HessianType == :SO
    σ += Hx_SO(h_3131, h_3231, h_3121, v[2:end], num_MO)
  else
    σ += Hx_SCI(EC, fock_MO, cMO, v[2:end], num_MO, D1)
  end
  newσ_hb = [[g'*v[2:end].* alpha];σ]
  newσ_hb[2:end] .+= g .* v[1] .*alpha
  return newσ_hb
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

function davidson(EC::ECInfo, v::Vector, N::Integer, n_max::Integer, thres::Number,  num_MO::Vector{Int64},
  h_block::NTuple{10, Matrix{Float64}}, g::Vector, alpha::Number, initVecType::Symbol, 
  fock_MO::Matrix, cMO::Matrix, HessianType::Symbol, D1::Matrix, convTrack::Bool=false)
  V = zeros(N,n_max)
  σ = zeros(N,n_max)
  h = zeros(n_max,n_max)
  ac = zeros(n_max)
  λ = zeros(n_max)
  n_MO = size(cMO,2)
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  occv = setdiff(1:n_MO, EC.space['o']) # to be modified
  eigvec_index = 1
  pick_vec = 6
  converged = false
  n_2, n_1o, n_v = num_MO
  h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232 = h_block
  if HessianType == :SO
    H0_hb = [[0.];diag(h_2121);diag(h_3131);diag(h_2222);diag(h_3232)]
  else
    h3131_SCIdiag = zeros(n_v,n_2)
    h3131_SCIdiag .+= 4.0 * diag(fock_MO[occv,occv])
    h3131_SCIdiag .-= 4.0 * reshape(diag(fock_MO[occ2,occ2]), 1, n_2)
    h3131_SCIdiag = reshape(h3131_SCIdiag, n_v*n_2)
    H0_hb = [[0.];diag(h_2121);h3131_SCIdiag;diag(h_2222);diag(h_3232)]
  end
  initGuessIndex = findmax(abs.(H0_hb))[2]
  numInitialVectors = 0 

  if initVecType == :RANDOM
    v = rand(size(v,1))
    v = v ./ norm(v)
    numInitialVectors = 1
    V[:,1] = v
  elseif initVecType == :INHERIT
    v = v ./ norm(v)
    numInitialVectors = 1
    V[:,1] = v
  elseif initVecType == :GRADIENT_SET
    V[1,1] = 1.0
    g_r = g + rand(size(g,1)) .* 0.02 .- 0.01
    v = [[0.];g_r] ./ norm(g_r)
    V[:,2] = v
    σ[:,1] = H_multiply(EC, fock_MO, cMO, D1, V[:,1], num_MO, g, alpha, h_block, HessianType)
    numInitialVectors = 2
  elseif initVecType == :GRADIENT_SETPLUS
    V[1,1] = 1.0
    g_r = g 
    v = [[0.];g_r] ./ norm(g_r)
    V[:,2] = v
    σ[:,1] = H_multiply(EC, fock_MO, cMO, D1, V[:,1], num_MO, g, alpha, h_block, HessianType)
    σ[:,2] = H_multiply(EC, fock_MO, cMO, D1, V[:,2], num_MO, g, alpha, h_block, HessianType)
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
  
  davCounti = 0
  for i in numInitialVectors+1:n_max
    davCounti += 1
    # blockwise H * v
    newσ_hb = H_multiply(EC, fock_MO, cMO, D1, v, num_MO, g, alpha, h_block, HessianType)
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
  return λ[eigvec_index], v, converged, davCounti
end

"""
    λTuning(EC::ECInfo, trust::Number, maxit4alpha::Integer, λmax::Number, λ::Number, h::Matrix, g::Vector)

Find the rotation parameters as the vector x in trust region,
tuning λ with the norm of x in the iterations.
Return λ and x.
"""
function λTuning(EC::ECInfo, trust::Number, maxit4alpha::Integer, alphamax::Number, alpha::Number, g::Vector, vec::Vector, num_MO::Vector{Int64}, 
  h_block::NTuple{10, Matrix{Float64}}, initVecType::Symbol, fock_MO::Matrix, cMO::Matrix, HessianType::Symbol, D1::Matrix, reject::Bool)
  davCount = 0
  N_rk = (num_MO[2]+num_MO[3]) * (num_MO[1]+num_MO[2])
  x = zeros(N_rk)
  alphal = 1.0
  alphar = alphamax
  xalphal = -1.0
  xalphar = -1.0
  micro_converged = false
  davItMax = 200 # for davidson eigenvalue solving algorithm
  davError = 1e-7
  bisecdamp = EC.options.scf.bisecdamp
  γ =  0.1 # gradient scaling factor for micro-iteration accuracy
  davError = γ * norm(g)
  alphaSearchIt = 0
  alphas = Array{Float64}(undef,0)
  trustTune = true
  # alpha tuning loop (micro loop)
  for it=1:maxit4alpha
    push!(alphas, alpha)
    @timeit "davidson" val, vec, converged, davCounti = davidson(EC, vec, N_rk+1, davItMax, davError, num_MO, h_block, g, alpha, initVecType, fock_MO, cMO, HessianType, D1)
    davCount += davCounti
    while !converged
      davItMax += 50
      println("Davidson max iteration number increased to ", davItMax)
      @timeit "davidson" val, vec, converged, davCounti = davidson(EC, vec, N_rk+1, davItMax, davError, num_MO, h_block, g, alpha, initVecType, fock_MO, cMO, HessianType, D1)
      davCount += davCounti
    end
    x = vec[2:end] ./ (vec[1] * alpha)
    # check if square of norm of x in trust region (0.8*trust ~ trust)
    sumx2 = sqrt(sum(x.^2))
    # sumx2 = sum(x.^2)
    #println("trust: ", trust, " sumx2: ", sumx2)
    if sumx2 > trust
      alphal = alpha
      xalphal = sumx2
    elseif sumx2 < 0.8*trust
      alphar = alpha
      xalphar = sumx2
    else
      micro_converged = true
      alphaSearchIt = it
      break
    end
    if alphar ≈ alphal
      alpha = alphal
      micro_converged = true
      alphaSearchIt = it
      break
    end
    # it means at the beginning of the search, if the alphal didn't move, as 1.0 
    # and alpha needs to be smaller, and meanwhile the alpharight is less than alphal + 0.1, as 1.1 
    # then alpha should be set to 1
    if xalphal < 0 && alpha == alphar && (alphar - alphal) < 0.1
      alpha = alphal
    elseif xalphal > 0 && xalphar > 0
      # line-search
      alpha = ((xalphal-trust)*alphar - (xalphar-trust)*alphal) / (xalphal - xalphar)
    else
      # damped geometric mean
      alpha = exp(log(alphal) + log(alphar/alphal) * bisecdamp)
    end
  end
  if !micro_converged
    alphaSearchIt = maxit4alpha
    println("micro NOT converged")
  end
  if alpha ≈ 1.0 && !reject
    trustTune = false
  end
  println(alphas)
  return alpha, x, vec, davCount, alphaSearchIt, trustTune
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
function checkE_modifyTrust(E, E_former, E_2o, trust, trustTune::Bool)
  energy_diff = E - E_former
  energy_quotient = energy_diff / E_2o
  # modify the trust region
  reject = false
  if energy_quotient < 0.0 || E_2o > 0.0
    if trustTune
      trust = 0.7 * trust
    end
    reject = true
  elseif energy_quotient < 0.25
    if trustTune
      trust = 0.7 * trust
    end
  elseif energy_quotient > 0.75
    if trustTune
      trust = 1.2 * trust
    end
  end
  return reject, trust
end

function print_initial(Enuc::Float64, HessianType::Symbol)
  println("Enuc ", Enuc)
  if HessianType == :SO
    HessianTypeString = "Second Order Approximation"
  elseif HessianType == :SCI
    HessianTypeString = "Super CI (First Order Approximation)"
  elseif HessianType == :SO_SCI
    HessianTypeString = "Combined Second Order and Super CI Approximation"
  end  
  println("Hessian Type: ", HessianTypeString)
end

"""
    dfmcscf(EC::ECInfo; direct=false, guess=:SAD, IterMax=64, maxit4alpha=100, HessianType=:SO)

Main body of Density-Fitted Multi-Configurational Self-Consistent-Field method
"""
function dfmcscf(EC::ECInfo; direct=false)
  guess = EC.options.scf.guess
  IterMax = EC.options.scf.IterMax
  maxit4alpha = EC.options.scf.maxit4alpha
  HessianType = EC.options.scf.HessianType
  initVecType = EC.options.scf.initVecType
  print_info("DF-MCSCF")
  setup_space_ms!(EC)
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  print_initial(Enuc, HessianType)

  #load info
  sao = load(EC,"S_AA")
  nAO = size(sao,2) # number of atomic orbitals
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified  
  occ1o = setdiff(EC.space['o'],occ2) # to be modified
  occv = setdiff(1:nAO, EC.space['o']) # to be modified
  n_2 = size(occ2,1)
  n_1o = size(occ1o, 1)
  n_v = size(occv,1)
  num_MO = [n_2,n_1o,n_v]
  N_rk = (n_1o+n_v) * (n_2+n_1o)

  # initial guess for inherit initial guess
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
  trust = 0.632
  λ = 500.0
  t0 = time_ns()
  x = []
  davCount = 0
  alphaSearchIt = 0
  trustTune = true
  E_2o = 0.0
  E = 0.0
  prev_cMO = deepcopy(cMO)
  μνL = load(EC,"AAL")
  prev_fock_MO = zeros(nAO,nAO)
  prev_fockClosed_MO = zeros(nAO,nAO)
  μjL = zeros(nAO,n_2,size(μνL,3))
  μuL = zeros(nAO,n_1o,size(μνL,3))
  # macro loop, g and h updated
  while iteration_times < IterMax
    # calc energy E with updated cMO
    @timeit "μjL" @tensoropt μjL[μ,j,L] = μνL[μ,ν,L] * cMO[:,occ2][ν,j]
    @timeit "μuL" @tensoropt μuL[μ,u,L] = μνL[μ,ν,L] * cMO[:,occ1o][ν,u]
    @timeit "fock calc" fock_MO, fockClosed_MO= dffockCAS(μνL, μjL, μuL, EC, cMO, D1)
    E_former = E
    E = calc_realE(μuL, EC, fockClosed_MO, D1, D2, cMO)
    # check if reject the update and tune trust
    if iteration_times > 0
      tt = (time_ns() - t0)/10^9
      @printf "%3i %12.8f %12.8f %12.8f %8.2f %12.6f %12.6f %12.6f %3i %3i \n" iteration_times E+Enuc E-E_former norm(g) tt trust sqrt(sum(x.^2)) λ davCount alphaSearchIt
      reject, trust = checkE_modifyTrust(E, E_former, E_2o, trust, trustTune)
      if reject
        iteration_times -= 1
        cMO = prev_cMO
        @timeit "μjL" @tensoropt μjL[μ,j,L] = μνL[μ,ν,L] * cMO[:,occ2][ν,j]
        @timeit "μuL" @tensoropt μuL[μ,u,L] = μνL[μ,ν,L] * cMO[:,occ1o][ν,u]
        fock_MO = deepcopy(prev_fock_MO)
        fockClosed_MO = deepcopy(prev_fockClosed_MO)
        E = E_former
        inherit_large = false
      elseif E_former - E < 1e-7 && E < E_former && norm(g) < 5e-3
        break
      end
    else
      println("Initial energy: ", E+Enuc)
      println("Iter     Energy      DE           norm(g)       Time      trust        sumx2        α      microIter")
    end
    prev_fock_MO = deepcopy(fock_MO)
    prev_fockClosed_MO = deepcopy(fockClosed_MO)
    iteration_times += 1
    # calc g and h with updated cMO
    A = dfACAS(μuL, EC,cMO,D1,D2,fock_MO,fockClosed_MO)
    g = calc_g(A, EC)
    if norm(g) < 1e-5
      break
    end
    if HessianType == :SO
      @timeit "abL" @tensoropt abL[a,b,L] := μνL[μ,ν,L] * cMO[:,occv][μ,a] * cMO[:,occv][ν,b]
      @timeit "h calc new" h_block = calc_h_SO(μjL, μuL, abL, EC, cMO, D1, D2, fock_MO, fockClosed_MO, A)
      abL = 0
    else
      @timeit "h calc new" h_block = calc_h_SCI(μjL, μuL, EC, cMO, D1, D2, fock_MO, fockClosed_MO, A, HessianType)
    end

    # λ tuning loop (micro loop)
    λmax = 1000.0
    if inherit_large == false
      vec = rand(N_rk+1)
      vec = vec./norm(vec)
      inherit_large == true
    end
    λ, x, vec, davCount, alphaSearchIt, trustTune = λTuning(EC, trust, maxit4alpha, λmax, λ, g, vec, num_MO, h_block, initVecType, fock_MO, cMO, HessianType, D1, reject)

    # calc 2nd order perturbation energy
    h_2121, h_3121, h_3131, h_2221, h_2231, h_2222, h_3221, h_3231, h_3222, h_3232 = h_block
    σ = Hx_common(h_2121, h_2221, h_3221, h_2231, h_2222, h_3222, h_3232, x)
    if HessianType == :SO
      σ .+= Hx_SO(h_3131, h_3231, h_3121, x, num_MO)
    else
      σ .+= Hx_SCI(EC, fock_MO, cMO, x, num_MO, D1)
    end
    E_2o = sum(g .* x) + 0.5*(transpose(x) * σ)

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
  if iteration_times < IterMax
    println("Convergent!")
  else
    println("Not Convergent!")
  end
  delete_temporary_files!(EC)
  return E+Enuc, cMO
end
end #module
