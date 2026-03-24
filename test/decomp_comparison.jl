"""
Comparison test: symmetric matrix decomposition methods.

Tests three approaches:
1. Pivoted Cholesky (symmetric_pivoted_cholesky) — requires PSD
2. Direct SVD/Takagi — full matrix decomposition
3. QR-pivoted two-step (qr_pivoted_symmetric_decompose) — no PSD requirement

Run with: julia --project=. test/decomp_comparison.jl
"""

using LinearAlgebra
using Random

# Load ElemCo for the decomposition functions
using ElemCo
using ElemCo.DecompTools: symmetric_pivoted_cholesky, 
  qr_pivoted_symmetric_decompose,
  qr_pivoted_decompose,
  ldlt_pivoted_symmetric_decompose,
  orthogonalize

"""Direct SVD/Takagi decomposition (baseline) for comparison.
For the RI formula, this is equivalent to using ALL columns as pivots."""
function direct_svd_decompose(M::AbstractMatrix{T}, tol) where T
  F = svd(M)
  nB = count(s -> s > tol, F.S)
  nB = max(nB, 1)
  if T <: Complex
    # Takagi: M = U_T Σ U_T^T → L = U_T √Σ → L L^T = M
    A = F.U[:, 1:nB]
    B = F.Vt[1:nB, :]'
    phases = [conj(sum(A[:,k] .* B[:,k])) for k in 1:nB]
    L = A .* transpose(sqrt.(phases) .* sqrt.(F.S[1:nB]))
    neg_indices = Int[]
  else
    # Eigendecomposition for real symmetric: M = Q Λ Q^T
    # L_k = q_k * √|λ_k|; negative λ_k tracked in neg_indices
    E = eigen(Symmetric(M))
    keep = findall(e -> abs(e) > tol, E.values)
    nB = length(keep)
    vals = E.values[keep]
    vecs = E.vectors[:, keep]
    L = vecs .* sqrt.(abs.(vals))'
    neg_indices = findall(v -> v < 0, vals)
  end
  return L, nB, neg_indices
end

function recon_error(M, L, neg_indices=Int[])
  if isempty(neg_indices)
    M_recon = L * transpose(L)
  else
    signs = ones(eltype(L), size(L, 2))
    signs[neg_indices] .= -1
    M_recon = L * Diagonal(signs) * transpose(L)
  end
  return maximum(abs.(M - M_recon))
end

function test_case(name, M, tol; expect_cholesky_fail=false)
  n = size(M, 1)
  T = eltype(M)
  println("\n" * "="^60)
  println("Test: $name")
  println("  Size: $n×$n, Type: $T, tol: $tol")

  # Check symmetry
  sym_err = maximum(abs.(M - transpose(M)))
  println("  Symmetry error: $sym_err")

  # 1. Direct SVD/Takagi (baseline)
  L_svd, rank_svd, neg_svd = direct_svd_decompose(M, tol)
  err_svd = recon_error(M, L_svd, neg_svd)
  println("  Direct SVD/Takagi: rank=$rank_svd, neg=$(length(neg_svd)), max_error=$err_svd")

  # 2. Pivoted Cholesky (PSD only)
  if T <: Complex
    try
      L_chol, rank_chol = symmetric_pivoted_cholesky(M, tol)
      err_chol = recon_error(M, L_chol)
      println("  Pivoted Cholesky:  rank=$rank_chol, max_error=$err_chol")
      if expect_cholesky_fail
        println("  WARNING: Cholesky succeeded when failure was expected!")
      end
    catch e
      println("  Pivoted Cholesky:  FAILED ($e)")
      if !expect_cholesky_fail
        println("  WARNING: Unexpected Cholesky failure!")
      end
    end
  else
    try
      M_herm = Hermitian(M)
      CA = cholesky(M_herm, RowMaximum(), check=false, tol=tol)
      rank_chol = CA.rank
      K_mat = CA.U[1:rank_chol, invperm(CA.p)]'
      err_chol = recon_error(M, K_mat)
      println("  Pivoted Cholesky:  rank=$rank_chol, max_error=$err_chol")
      if expect_cholesky_fail
        println("  WARNING: Cholesky succeeded when failure was expected!")
      end
    catch e
      println("  Pivoted Cholesky:  FAILED ($e)")
      if !expect_cholesky_fail
        println("  WARNING: Unexpected Cholesky failure!")
      end
    end
  end

  # 3. QR-pivoted two-step (span-factor batched)
  L_qr, rank_qr, neg_qr = qr_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  err_qr = recon_error(M, L_qr, neg_qr)
  println("  QR two-step:      rank=$rank_qr, neg=$(length(neg_qr)), max_error=$err_qr")

  # 3b. QR decompose + orthogonalize (should give same result)
  L_qr2, rank_qr2, neg_qr2 = qr_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  ortho_qr, diag_qr = orthogonalize(L_qr2, neg_qr2)
  err_ortho_qr = recon_error(M, L_qr2, neg_qr2)
  println("  QR decompose:     rank=$rank_qr2, neg=$(length(neg_qr2)), ortho=$(size(ortho_qr)), max_error=$err_ortho_qr")

  # 4. LDLT-pivoted two-step (span-factor batched, works for indefinite)
  L_ldlt, rank_ldlt, neg_ldlt = ldlt_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  err_ldlt = recon_error(M, L_ldlt, neg_ldlt)
  println("  LDLT two-step:    rank=$rank_ldlt, neg=$(length(neg_ldlt)), max_error=$err_ldlt")

  # 4b. LDLT decompose + orthogonalize (should give same result)
  L_ldlt2, rank_ldlt2, neg_ldlt2 = ldlt_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  ortho_ldlt, diag_ldlt = orthogonalize(L_ldlt2, neg_ldlt2)
  err_ortho_ldlt = recon_error(M, L_ldlt2, neg_ldlt2)
  println("  LDLT decompose:   rank=$rank_ldlt2, neg=$(length(neg_ldlt2)), ortho=$(size(ortho_ldlt)), max_error=$err_ortho_ldlt")

  println("="^60)
end

# ─────────────────────────────────────────────────────────────
# Test matrices
# ─────────────────────────────────────────────────────────────
Random.seed!(42)
tol = 1e-8

# Test 1: Real symmetric PSD (low rank)
println("\n" * "#"^60)
println("# REAL SYMMETRIC MATRICES")
println("#"^60)

n = 100
r = 20  # rank
A = randn(n, r)
M_psd = A * A'  # real symmetric PSD, rank r
test_case("Real PSD (rank $r)", M_psd, tol)

# Test 2: Real symmetric PSD (full rank)
M_full = randn(n, n)
M_full = M_full * M_full'  # full rank PSD
test_case("Real PSD (full rank)", M_full, tol)

# Test 3: Real symmetric indefinite
D = Diagonal(vcat(ones(50), -ones(50)))
Q, _ = qr(randn(n, n))
Q = Matrix(Q)
M_indef = Q * D * Q'  # eigenvalues: 50 positive, 50 negative
test_case("Real indefinite (50+, 50-)", M_indef, tol; expect_cholesky_fail=true)

# Test 4: Real symmetric indefinite (low rank)
r_pos, r_neg = 10, 5
A_pos = randn(n, r_pos)
A_neg = randn(n, r_neg)
M_indef_lr = A_pos * A_pos' - A_neg * A_neg'  # rank ≤ r_pos + r_neg
test_case("Real indefinite low-rank ($r_pos+, $r_neg-)", M_indef_lr, tol; expect_cholesky_fail=true)

# Complex symmetric matrices
println("\n" * "#"^60)
println("# COMPLEX SYMMETRIC MATRICES")
println("#"^60)

# Test 5: Complex symmetric PSD-like (from Gram-like construction)
n = 80
r = 15
A_c = randn(ComplexF64, n, r)
M_cpsd = A_c * transpose(A_c)  # complex symmetric, M = A * A^T
test_case("Complex PSD-like (rank $r)", M_cpsd, tol)

# Test 6: Complex symmetric with mixed phases
# Create via Takagi: M = U Σ U^T with random unitary U
U_rand, _ = qr(randn(ComplexF64, n, n))
U_rand = Matrix(U_rand)
Sigma = Diagonal(vcat(10.0 .* rand(20), 0.01 .* rand(60)))
M_takagi = U_rand * Sigma * transpose(U_rand)  # complex symmetric
test_case("Complex symmetric (Takagi-constructed)", M_takagi, tol)

# Test 7: Complex symmetric with large dynamic range (ill-conditioned Cholesky)
Sigma_wide = Diagonal(10.0 .^ range(0, -12, length=n))
M_illcond = U_rand * Sigma_wide * transpose(U_rand)
test_case("Complex symmetric (ill-conditioned)", M_illcond, tol)

# Test 8: Complex symmetric — non-PSD in Hermitian sense but valid Takagi
# Construct a matrix where real parts of diagonals are negative
n = 50
r = 10
# Use phases that create negative real-part diagonals
B1 = (1.0 + 1.0im) * randn(ComplexF64, n, r)
M_neg_diag = B1 * transpose(B1)
# Check how many diagonals have negative real part
n_neg = count(real(M_neg_diag[i,i]) < 0 for i in 1:n)
test_case("Complex symmetric ($n_neg/$n neg. real diagonals)", M_neg_diag, tol;
          expect_cholesky_fail = n_neg > 0)

# ─────────────────────────────────────────────────────────────
# General (non-symmetric, rectangular) matrix tests
# ─────────────────────────────────────────────────────────────
println("\n" * "#"^60)
println("# GENERAL (NON-SYMMETRIC / RECTANGULAR) MATRICES")
println("#"^60)

function test_general(name, M, tol)
  m, n = size(M)
  T = eltype(M)
  println("\n" * "="^60)
  println("Test: $name")
  println("  Size: $m×$n, Type: $T, tol: $tol")

  result = qr_pivoted_decompose(M, tol; sigma=0.01)
  err = maximum(abs.(M - result.Q * result.R))
  ortho_err = maximum(abs.(result.Q' * result.Q - I(result.rank)))
  println("  QR general:      rank=$(result.rank), max_error=$err, ortho_error=$ortho_err")
  println("="^60)
end

# Test G1: Non-symmetric square, low rank
n = 100; r = 20
M_g1 = randn(n, r) * randn(r, n)
test_general("Non-symmetric square low-rank ($r)", M_g1, tol)

# Test G2: Tall rectangular, low rank
m, n2 = 150, 80
M_g2 = randn(m, r) * randn(r, n2)
test_general("Tall $(m)×$(n2) low-rank ($r)", M_g2, tol)

# Test G3: Wide rectangular, low rank
M_g3 = randn(n2, r) * randn(r, m)
test_general("Wide $(n2)×$(m) low-rank ($r)", M_g3, tol)

# Test G4: Full-rank square
M_g4 = randn(50, 50)
test_general("Full-rank 50×50", M_g4, tol)

# Test G5: Complex non-symmetric rectangular
M_g5 = randn(ComplexF64, m, r) * randn(ComplexF64, r, n2)
test_general("Complex $(m)×$(n2) low-rank ($r)", M_g5, tol)

# Test G6: Very wide matrix (m < n)
M_g6 = randn(10, r) * randn(r, 200)
test_general("Very wide 10×200 low-rank ($r)", M_g6, tol)

# Test G7: Very tall matrix (m >> n)
M_g7 = randn(200, r) * randn(r, 10)
test_general("Very tall 200×10 low-rank ($r)", M_g7, tol)

# ─────────────────────────────────────────────────────────────
# Numerical redundancy tests: gradual singular value decay
# ─────────────────────────────────────────────────────────────
println("\n" * "#"^60)
println("# NUMERICAL REDUNDANCY: GRADUAL SINGULAR VALUE DECAY")
println("#"^60)

"""
    make_symmetric_decay(n, decay_type; kwargs...) → M

Build an `n×n` real symmetric matrix with controlled singular value decay.

`decay_type` can be:
- `:exponential` — `σ_i = exp(-α * i)`, kwarg `α` (default 0.3)
- `:polynomial`  — `σ_i = (i+1)^{-p}`, kwarg `p` (default 2)
- `:linear`      — `σ_i` linearly from 1 to `smin`, kwarg `smin` (default 1e-12)
- `:step`        — first `k_big` values = 1, rest ∈ [gap, gap*ε], kwarg `k_big`, `gap` (default 1e-5)
"""
function make_symmetric_decay(n, decay_type; α=0.3, p=2.0, smin=1e-12, k_big=div(n,5), gap=1e-5)
  if decay_type == :exponential
    svals = [exp(-α * i) for i in 1:n]
  elseif decay_type == :polynomial
    svals = [(i + 1.0)^(-p) for i in 1:n]
  elseif decay_type == :linear
    svals = range(1.0, smin, length=n) |> collect
  elseif decay_type == :step
    svals = vcat(ones(k_big), range(gap, gap * 1e-7, length=n - k_big) |> collect)
  else
    error("Unknown decay type: $decay_type")
  end
  Q_mat, _ = qr(randn(n, n))
  Q_mat = Matrix(Q_mat)
  return Q_mat * Diagonal(svals) * Q_mat', svals
end

"""
    make_general_decay(m, n, decay_type; kwargs...) → M

Build an `m×n` real matrix with controlled singular value decay.

Same decay types as `make_symmetric_decay`.
"""
function make_general_decay(m, n, decay_type; α=0.3, p=2.0, smin=1e-12, k_big=div(min(m,n),5), gap=1e-5)
  r = min(m, n)
  if decay_type == :exponential
    svals = [exp(-α * i) for i in 1:r]
  elseif decay_type == :polynomial
    svals = [(i + 1.0)^(-p) for i in 1:r]
  elseif decay_type == :linear
    svals = range(1.0, smin, length=r) |> collect
  elseif decay_type == :step
    svals = vcat(ones(k_big), range(gap, gap * 1e-7, length=r - k_big) |> collect)
  else
    error("Unknown decay type: $decay_type")
  end
  U_mat, _ = qr(randn(m, m))
  V_mat, _ = qr(randn(n, n))
  U_mat = Matrix(U_mat)
  V_mat = Matrix(V_mat)
  return U_mat[:, 1:r] * Diagonal(svals) * V_mat[:, 1:r]', svals
end

function test_decay_symmetric(name, M, svals, tol)
  n = size(M, 1)
  # True rank at this tolerance
  true_rank = count(s -> s > tol, svals)

  println("\n" * "="^60)
  println("Test: $name")
  println("  Size: $n×$n, tol: $tol, true rank at tol: $true_rank")
  println("  σ_1=$(svals[1]), σ_$(true_rank)=$(svals[true_rank]), σ_$(min(true_rank+1,n))=$(svals[min(true_rank+1,n)])")

  # QR symmetric
  L_qr, rank_qr, neg_qr = qr_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  err_qr = recon_error(M, L_qr, neg_qr)
  println("  QR symmetric:  rank=$rank_qr, max_error=$err_qr")

  # LDLT symmetric
  L_ldlt, rank_ldlt, neg_ldlt = ldlt_pivoted_symmetric_decompose(M, tol; sigma=0.01)
  err_ldlt = recon_error(M, L_ldlt, neg_ldlt)
  println("  LDLT symmetric: rank=$rank_ldlt, max_error=$err_ldlt")

  # QR general (applied to symmetric matrix for comparison)
  result_gen = qr_pivoted_decompose(M, tol; sigma=0.01)
  err_gen = maximum(abs.(M - result_gen.Q * result_gen.R))
  println("  QR general:    rank=$(result_gen.rank), max_error=$err_gen")

  println("="^60)
end

function test_decay_general(name, M, svals, tol)
  m, n = size(M)
  r = min(m, n)
  true_rank = count(s -> s > tol, svals)

  println("\n" * "="^60)
  println("Test: $name")
  println("  Size: $m×$n, tol: $tol, true rank at tol: $true_rank")
  println("  σ_1=$(svals[1]), σ_$(true_rank)=$(svals[true_rank]), σ_$(min(true_rank+1,r))=$(svals[min(true_rank+1,r)])")

  result = qr_pivoted_decompose(M, tol; sigma=0.01)
  err = maximum(abs.(M - result.Q * result.R))
  ortho_err = maximum(abs.(result.Q' * result.Q - I(result.rank)))
  println("  QR general:    rank=$(result.rank), max_error=$err, ortho_error=$ortho_err")

  println("="^60)
end

tol = 1e-6
# ── Symmetric matrices with decay ──
for n in [100, 500, 1000]
  println("\n" * "-"^60)
  println("  n = $n")
  println("-"^60)

  M_exp, s_exp = make_symmetric_decay(n, :exponential; α=0.3)
  test_decay_symmetric("Symmetric exponential decay (n=$n, α=0.3)", M_exp, s_exp, tol)

  M_poly, s_poly = make_symmetric_decay(n, :polynomial; p=2.0)
  test_decay_symmetric("Symmetric polynomial decay (n=$n, p=2)", M_poly, s_poly, tol)

  M_lin, s_lin = make_symmetric_decay(n, :linear; smin=1e-12)
  test_decay_symmetric("Symmetric linear decay (n=$n)", M_lin, s_lin, tol)

  M_step, s_step = make_symmetric_decay(n, :step; k_big=div(n, 5), gap=1e-5)
  test_decay_symmetric("Symmetric step decay (n=$n, k=$(div(n,5)))", M_step, s_step, tol)
end

# ── General rectangular matrices with decay ──
for (m, n) in [(200, 100), (100, 500), (500, 500), (1000, 200)]
  println("\n" * "-"^60)
  println("  m×n = $m×$n")
  println("-"^60)

  M_exp, s_exp = make_general_decay(m, n, :exponential; α=0.3)
  test_decay_general("General exponential decay ($m×$n, α=0.3)", M_exp, s_exp, tol)

  M_poly, s_poly = make_general_decay(m, n, :polynomial; p=2.0)
  test_decay_general("General polynomial decay ($m×$n, p=2)", M_poly, s_poly, tol)

  M_lin, s_lin = make_general_decay(m, n, :linear; smin=1e-12)
  test_decay_general("General linear decay ($m×$n)", M_lin, s_lin, tol)

  M_step, s_step = make_general_decay(m, n, :step; gap=1e-5)
  test_decay_general("General step decay ($m×$n)", M_step, s_step, tol)
end

println("\n\nAll comparison tests completed.")
