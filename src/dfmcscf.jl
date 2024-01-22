module DFMCSCF
using LinearAlgebra, TensorOperations, Printf
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
save in "AcL" and "AaL" on disk. 
"""
function projDenFitInt(EC::ECInfo, cMO::Matrix)
  μνL = load(EC,"AAL")
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  CMO2 = cMO[:,occ2]
  CMOa = cMO[:,occ1o] # to be modified
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
  @tensoropt μjL[μ,j,L] := μνL[μ,ν,L] * CMO2[ν,j]
  save!(EC,"AcL",μjL)
  @tensoropt μuL[μ,u,L] := μνL[μ,ν,L] * CMOa[ν,u]
  save!(EC,"AaL",μuL)
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
  μνL = load(EC,"AAL")
  μjL = load(EC,"AcL")
  μuL = load(EC,"AaL")

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
  μuL = load(EC,"AaL")
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
  occv = setdiff(1:size(A,1), EC.space['o']) # to be modified
  μνL = load(EC,"AAL")
  μjL = load(EC,"AcL")
  μuL = load(EC,"AaL")

  # Gij
  @tensoropt pjL[p,j,L] := μjL[μ,j,L] * cMO[μ,p] # to transfer the first index from atomic basis to molecular basis
  @tensoropt Gij[r,s,i,j] := 8 * pjL[r,i,L] * pjL[s,j,L]
  @tensoropt Gij[r,s,i,j] -= 2 * pjL[s,i,L] * pjL[r,j,L]
  ijL = pjL[occ2,:,:]
  @tensoropt pqL[p,q,L] := μνL[μ,ν,L] * cMO[μ,p] * cMO[ν,q]
  @tensoropt Gij[r,s,i,j] -= 2 * ijL[i,j,L] * pqL[r,s,L]
  Iij = 1.0 * Matrix(I, length(occ2), length(occ2))
  @tensoropt Gij[r,s,i,j] += 2 * fock[μ,ν] * cMO[μ,r] * cMO[ν,s] * Iij[i,j] # lower cost?

  # Gtj
  @tensoropt puL[p,u,L] := μuL[μ,u,L] * cMO[μ,p] #transfer from atomic basis to molecular basis
  @tensoropt testStuff[r,s,v,j] := puL[r,v,L] * pjL[s,j,L]
  @tensoropt multiplier[r,s,v,j] := 4 * puL[r,v,L] * pjL[s,j,L]
  @tensoropt multiplier[r,s,v,j] -= puL[s,v,L] * pjL[r,j,L]
  tjL = pjL[occ1o,:,:]
  @tensoropt multiplier[r,s,v,j] -= pqL[r,s,L] * tjL[v,j,L]
  @tensoropt Gtj[r,s,t,j] := multiplier[r,s,v,j] * D1[t,v]
  
  # Gtu 
  @tensoropt Gtu[r,s,t,u] := fockClosed[μ,ν] * cMO[μ,r] * cMO[ν,s] * D1[t,u]
  tuL = pqL[occ1o, occ1o, :]
  @tensoropt Gtu[r,s,t,u] += pqL[r,s,L] * (tuL[v,w,L] * D2[t,u,v,w])
  @tensoropt Gtu[r,s,t,u] += 2 * (puL[r,v,L] * puL[s,w,L]) * D2[t,v,u,w]

  # G
  n_AO = size(cMO,1)
  G = zeros((n_AO,n_AO,n_AO,n_AO))
  G[:,:,occ2,occ2] = Gij
  G[:,:,occ1o,occ2] = Gtj
  G[:,:,occ2,occ1o] = permutedims(Gtj, [2,1,4,3])
  G[:,:,occ1o,occ1o] = Gtu

  # calc h with G
  # matters can be saved here
  # 4 * O(N^4)
  I_kl = 1.0 * Matrix(I, n_AO, n_AO)
  @tensoropt h[r,k,s,l] := 2 * G[r,s,k,l] - I_kl[k,l] * (A[r,s] + A[s,r])
  @tensoropt h[r,k,s,l] += 2 * G[k,l,r,s] - I_kl[r,s] * (A[k,l] + A[l,k])
  @tensoropt h[r,k,s,l] -= 2 * G[k,s,r,l] - I_kl[r,l] * (A[k,s] + A[s,k])
  @tensoropt h[r,k,s,l] -= 2 * G[r,l,k,s] - I_kl[k,s] * (A[r,l] + A[l,r])
  h_rk_sl = h[[occ1o;occv],[occ2;occ1o],[occ1o;occv],[occ2;occ1o]]
  d = size(h_rk_sl,1) * size(h_rk_sl,2)
  h_rk_sl = reshape(h_rk_sl, d, d)

  #save!(EC,"h_rk_sl",h_rk_sl)
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
  hsmall = load(EC,"h_AA")
  CMO2 = cMO[:,occ2] 
  CMOa = cMO[:,occ1o] 
  μνL = load(EC,"AAL")
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
function davidson(H::Matrix, N::Integer, n::Integer, thres::Number, convTrack::Bool=false)
  V = zeros(N,n)
  σ = zeros(N,n)
  h = zeros(n,n)
  v = rand(N)
  v = v./norm(v)
  V[:,1] = v
  ac = zeros(n)
  H0 = diag(H)
  λ = zeros(n)
  eigvec_index = 1
  pick_vec = 50
  converged = false
  for i in 2:n
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
      println("Iter ", i, " converged!")
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
function λTuning(trust::Number, maxit::Integer, λmax::Number, λ::Number, h::Matrix, g::Vector)
  x = zeros(size(h,1))
  λl = 1.0
  λr = λmax
  micro_converged = false
  N_rk = size(h,1)
  davItMax = 100 # for davidson eigenvalue solving algorithm
  davError = 1e-7
  # λ tuning loop (micro loop)
  for it=1:maxit
    # calc x
    W = zeros(N_rk+1, N_rk+1) # workng matrix W
    W[1, 2:N_rk+1] = g
    W[2:N_rk+1, 1] = g
    W[2:N_rk+1,2:N_rk+1] = h./λ
    W = Matrix(Hermitian(W))
    vec = zeros(N_rk+1)
    if N_rk < 600
      vals, vecs = eigen(W)
      vec = vecs[:,1]
    else
      val, vec, converged = davidson(W, N_rk+1, davItMax, davError)
      while !converged
        davItMax += 50
        println("Davidson max iteration number increased to ", davItMax)
        val, vec, converged = davidson(W, N_rk+1, davItMax, davError)
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
  return λ, x
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
    println("REJECT the update of coefficients, new trust value: ", trust)
  elseif energy_quotient < 0.25
    trust = 0.7 * trust
  elseif energy_quotient > 0.75
    trust = 1.2 * trust
  end
  return reject, trust
end

"""
    dfmcscf(EC::ECInfo; direct = false, guess=:SAD, IterMax=50)

Main body of Density-Fitted Multi-Configurational Self-Consistent-Field method
"""
function dfmcscf(EC::ECInfo; direct=false, guess=:SAD, IterMax=50)
  print_info("DF-MCSCF")
  setup_space_ms!(EC)
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  sao = load(EC,"S_AA")
  nAO = size(sao,2) # number of atomic orbitals
  occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
  occ1o = setdiff(EC.space['o'],occ2)
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
  while norm(g) > 1e-6 && iteration_times < IterMax

    # calc g and h with updated cMO
    projDenFitInt(EC, cMO)
    fock, fockClosed = dffockCAS(EC,cMO,D1)
    A = dfACAS(EC,cMO,D1,D2,fock,fockClosed)
    g = calc_g(A, EC)
    h = calc_h(EC, cMO, D1, D2, fock, fockClosed, A)
    #println("norm of g: ", norm(g))
    
    # λ tuning loop (micro loop)
    λmax = 1000.0
    maxit = 100
    λ, x = λTuning(trust, maxit, λmax, λ, h, g)
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
    end

    iteration_times += 1
    E_former = E
  end
  if iteration_times < IterMax
    println("Convergent!")
  else
    println("Not Convergent!")
  end
  delete_temporary_files!(EC)
  return E_former+Enuc, cMO
end
end #module
