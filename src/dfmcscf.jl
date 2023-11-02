module DFMCSCF
using LinearAlgebra, TensorOperations, Printf, TimerOutputs
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.DIIS
using ..ElemCo.TensorTools
using ..ElemCo.DFHF

export dfmcscf
export davidson

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
  @tensoropt D2[t,u,v,w] := D1[t,u]*D1[v,w] - D1[t,w]*D1[v,u]
  return D1, D2
end

"""
    projDenFitInt(EC::ECInfo, cMO::Matrix)

Read the μνL density fitting integral, 
project to μjL and μuL with the coefficients cMO, 
j -> doubly occupied orbital, u -> active orbital, 
save in "mudL" and "muaL" on disk. 
"""
function projDenFitInt(EC::ECInfo, cMO::Matrix)
  μνL = load(EC,"munuL")
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  @tensoropt μjL[μ,j,L] := μνL[μ,ν,L] * CMO2[ν,j]
  save(EC,"mudL",μjL)
  @tensoropt μuL[μ,u,L] := μνL[μ,ν,L] * CMOa[ν,u]
  save(EC,"muaL",μuL)
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
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  μνL = load(EC,"munuL")
  μjL = load(EC,"mudL")
  μuL = load(EC,"muaL")

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
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  @tensoropt g[r,s] := A[r,s] - A[s,r]
  occv = setdiff(1:size(A,1), EC.space['o']) # to be modified
  grk = g[[occ1o;occv],[occ2;occ1o]] # to be modified
  grk = reshape(grk, size(grk,1) * size(grk,2))
  return grk
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
  @tensoropt pjL[p,j,L] := μjL[μ,j,L] * cMO[μ,p] # to transfer the first index from atomic basis to molecular basis
  @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] = 8 * pjL[r,i,L] * pjL[s,j,L]
  @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * pjL[s,i,L] * pjL[r,j,L]
  ijL = pjL[occ2,:,:]
  @tensoropt pqL[p,q,L] := μνL[μ,ν,L] * cMO[μ,p] * cMO[ν,q]
  @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] -= 2 * ijL[i,j,L] * pqL[r,s,L]
  Iij = 1.0 * Matrix(I, length(occ2), length(occ2))
  @tensoropt G[:,:,1:n_2,1:n_2][r,s,i,j] += 2 * fock[μ,ν] * cMO[μ,r] * cMO[ν,s] * Iij[i,j] 

  # Gtj
  @tensoropt puL[p,u,L] := μuL[μ,u,L] * cMO[μ,p] #transfer from atomic basis to molecular basis
  @tensoropt testStuff[r,s,v,j] := puL[r,v,L] * pjL[s,j,L]
  @tensoropt multiplier[r,s,v,j] := 4 * puL[r,v,L] * pjL[s,j,L]
  @tensoropt multiplier[r,s,v,j] -= puL[s,v,L] * pjL[r,j,L]
  tjL = pjL[occ1o,:,:]
  @tensoropt multiplier[r,s,v,j] -= pqL[r,s,L] * tjL[v,j,L]
  @tensoropt G[:,:,n_2+1:n_occ,1:n_2][r,s,t,j] = multiplier[r,s,v,j] * D1[t,v]

  # Gtu 
  @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] = fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s] * D1[t,u]
  tuL = pqL[occ1o, occ1o, :]
  @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += pqL[r,s,L] * (tuL[v,w,L] * D2[t,u,v,w])
  @tensoropt G[:,:,n_2+1:n_occ,n_2+1:n_occ][r,s,t,u] += 2 * (puL[r,v,L] * puL[s,w,L]) * D2[t,v,u,w]

  # Gjt
  G[:,:,1:n_2,n_2+1:n_occ] = permutedims(G[:,:,n_2+1:n_occ,1:n_2], [2,1,4,3])

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
  return h
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
  I_ab = 1.0 * Matrix(I, n_v, n_v)
  I_tu = 1.0 * Matrix(I, n_1o, n_1o)
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
  μνL = load(EC,"munuL")
  @tensoropt E = CMO2[μ,i]*(hsmall[μ,ν]+fockClosed[μ,ν])*CMO2[ν,i]
  @tensoropt fockClosed_MO[t,u] := fockClosed[μ,ν] * CMOa[μ,t] *CMOa[ν,u]
  E += sum(fockClosed_MO .* D1)
  @tensoropt tuL[t,u,L] := μνL[p,q,L] * CMOa[p,t] * CMOa[q,u]
  @tensoropt tuvw[t,u,v,w] := tuL[t,u,L] * tuL[v,w,L]
  E += 0.5 * sum(D2 .* tuvw)
  return E
end

"""
    davidson(H::Matrix, N::Integer, n::Integer, thres::Number, convTrack::Bool=false)

Calculate one of the eigenvalues and corresponding eigenvector of the matrix H
(usually the lowest eigenvalue), 
N is the size of the matrix H, 
n is the maximal size of projected matrix, 
thres is the criterion of convergence, 
convTrack is to decide whether the tracking of eigenvectors is used
"""
function davidson(H::Matrix, v::Vector, N::Integer, n::Integer, thres::Number, convTrack::Bool=false)
  V = zeros(N,n)
  σ = zeros(N,n)
  h = zeros(n,n)
  ac = zeros(n)
  H0 = diag(H)
  λ = zeros(n)
  eigvec_index = 1
  pick_vec = 1
  converged = false

  numInitialVectors = 0

  # random initial guess
  # v = rand(size(v,1))
  # numInitialVectors = 1

  # inherit a initial vector from last Davidson procedure
  v = v ./ norm(v)

  numInitialVectors = 1
  V[:,1] = v

  # a special set of initial vectors guess
  # V[1,1] = 1.0
  # b1 = H[:,1]
  # V[:,2] = b1 ./norm(b1)
  # h = V' * H * V
  # V[21,3] = 1.0
  # v = V[:,3]
  # numInitialVectors = 3
   
  for i in numInitialVectors+1:n
    newσ = H * v
    σ[:,i-1] = newσ
    newh = V' * newσ
    h[:,i-1] = newh
    h[i-1,:] = newh
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
    v = -1.0 ./ (H0 .- λ[eigvec_index]) .* r
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
function λTuning(trust::Number, maxit::Integer, λmax::Number, λ::Number, h::Matrix, g::Vector, vec::Vector)
  x = zeros(size(h,1))
  λl = 1.0
  λr = λmax
  micro_converged = false
  N_rk = size(h,1)
  davItMax = 100 # for davidson eigenvalue solving algorithm
  davError = 1e-7
  γ =  0.1 # gradient scaling factor for micro-iteration accuracy
  davError = γ * norm(g)
  #vec = rand(N_rk+1)
  #vec = vec ./ norm(vec)
  # λ tuning loop (micro loop)
  for it=1:maxit
    # calc x
    W = zeros(N_rk+1, N_rk+1) # workng matrix W
    W[1, 2:N_rk+1] = g
    W[2:N_rk+1, 1] = g
    W[2:N_rk+1,2:N_rk+1] = h./λ
    W = Matrix(Hermitian(W))
    if N_rk < 6
      vals, vecs = eigen(W)
      vec = vecs[:,1]
    else
      val, vec, converged = davidson(W, vec, N_rk+1, davItMax, davError)
      while !converged
        davItMax += 50
        println("Davidson max iteration number increased to ", davItMax)
        val, vec, converged = davidson(W, vec, N_rk+1, davItMax, davError)
      end
    end
    x = vec[2:end] ./ (vec[1]*λ)
    # check if square of norm of x in trust region (0.8*trust ~ trust)
    sumx2 = (1/vec[1]^2 - 1) / λ^2
    if sumx2 > trust
      λl = λ
    elseif sumx2 < 0.8*trust
      λr = λ
    else
      micro_converged = true
      break
    end
    if λr ≈ λl # norm of x too small
      micro_converged = true
      break
    end
    λ = (λl + λr) / 2
  end
  if !micro_converged
    println("micro NOT converged")
  end
  return λ, x, vec
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
  R = zeros(N_MO,N_MO)
  R_sub = reshape(x, N_MO-size(occ2,1), size(occ1o,1)+size(occ2,1))
  R[[occ1o;occv],[occ2;occ1o]] = R_sub
  R[[occ2;occ1o],[occ1o;occv]] = -1.0 .* transpose(R_sub)
  U = 1.0 * Matrix(I,N_MO,N_MO) + R
  U = U + 1/2 .* R*R + 1/6 .* R*R*R
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
  if energy_quotient < 0.0
    trust = 0.7 * trust
    reject = true
    #println("REJECT the update of coefficients, new trust value: ", trust)
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
function dfmcscf(EC::ECInfo; direct = false, guess = GUESS_SAD, IterMax=50)
  Enuc = generate_integrals(EC; save3idx=!direct)
  sao = load(EC,"sao")
  nAO = size(sao,2) # number of atomic orbitals
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  N_rk = (nAO - size(occ2,1)) * (size(occ1o,1)+size(occ2,1))
  vec = rand(N_rk+1)
  vec = vec ./ norm(vec)
  inherit_large = true
  if size(occ1o,1) == 0
    error("NO ACTIVE ORBITALS, PLEASE USE DFHF")
  end

  # cMO and density matrices initialization
  cMO = guess_orb(EC,guess)
  D1, D2 = denMatCreate(EC)

  # calc initial energy
  projDenFitInt(EC, cMO)
  fock, fockClosed = dffockCAS(EC,cMO,D1)
  E0 = calc_realE(EC, fockClosed, D1, D2, cMO)
  println("Enuc ", Enuc)
  println("Initial energy: ", E0+Enuc)

  # macro loop parameters
  iteration_times = 1
  g = [1]
  E_former = E0
  trust = 0.4
  λ = 500.0

  # macro loop, g and h updated
  while norm(g) > 2e-6 && iteration_times < IterMax

    # calc g and h with updated cMO
    projDenFitInt(EC, cMO)
    fock, fockClosed = dffockCAS(EC,cMO,D1)
    A = dfACAS(EC,cMO,D1,D2,fock,fockClosed)
    g = calc_g(A, EC)
    h = calc_h(EC, cMO, D1, D2, fock, fockClosed, A)
    #h = calc_h_SCI(EC, cMO, fock, D1, D2, h_SO)
    #h = calc_h_combined(EC, cMO, fock, D1, D2, h_SO)
    println("norm of g: ", norm(g))
    
    # λ tuning loop (micro loop)
    λmax = 1000.0
    maxit = 100
    if inherit_large == false
      vec = rand(N_rk+1)
      vec = vec./norm(vec)
      inherit_large == true
    end
    λ, x, vec = λTuning(trust, maxit, λmax, λ, h, g, vec)
    #println("square of the norm of x: ", sum(x.^2))

    # calc 2nd order perturbation energy
    E_2o = sum(g .* x) + 0.5*(transpose(x) * h * x)
    #println("2nd order perturbation energy difference: ", E_2o)
    
    # calc rotation matrix U
    U = calc_U(EC, nAO, x)
    #println("difference between U and a real unitary matrix: ", sum((U'*U-I).^2))

    # update cMO with U
    prev_cMO = deepcopy(cMO)
    cMO = cMO*U

    # reorthogonalize molecular orbitals
    smo = cMO' * sao * cMO
    cMO = cMO * Hermitian(smo)^(-1/2)

    # calc energy E with updated cMO
    projDenFitInt(EC, cMO)
    fock, fockClosed = dffockCAS(EC,cMO,D1)
    E = calc_realE(EC, fockClosed, D1, D2, cMO)
    println("Iter ", iteration_times, " energy: ", E+Enuc)

    # check if reject the update and tune trust
    reject, trust = checkE_modifyTrust(E, E_former, E_2o, trust)
    if reject
      cMO = prev_cMO
      E = E_former
      iteration_times -= 1
      inherit_large = false
    end

    iteration_times += 1
    E_former = E
  end
  if iteration_times < IterMax
    println("Convergent!")
  else
    println("Not Convergent!")
  end
  return E_former+Enuc, cMO
end
end #module
