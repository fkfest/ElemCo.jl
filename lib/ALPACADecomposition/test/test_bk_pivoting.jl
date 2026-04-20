# ──────────────────────────────────────────────────────────────────
# Tests for Bunch-Kaufman 2×2 pivoting in ALPACA
# ──────────────────────────────────────────────────────────────────

# ── Real symmetric anti-symmetric (zero diagonal forces 2×2) ──

@testitem "BK: anti-symmetric matrix (zero diagonal)" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(901)

  # Anti-symmetric block embedded in a symmetric matrix:
  # Build a matrix with zero diagonal where all signal is off-diagonal,
  # forcing 2×2 pivoting for every pair.
  n = 6
  # A = Q * diag(λ) * Q' with eigenvalues [2, -2, 1, -1, 0.5, -0.5]
  # This has zero trace (sum of eigenvalues = 0) and indefinite structure
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [2.0, -2.0, 1.0, -1.0, 0.5, -0.5]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric
  @test length(result.neg_indices) >= 3  # at least 3 negative eigenvalues

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: pure anti-symmetric block" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  # 2×2 symmetric with zero diagonal: [[0, a], [a, 0]]
  # Diagonal is exactly zero → must use 2×2 pivot
  a = 3.0
  A = [0.0  a;
       a    0.0]

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  # eigenvalues are ±a, so both pivots found
  @test length(result.pivot_indices) == 2
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "BK: larger zero-diagonal symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(902)

  # Block-diagonal symmetric with zero diagonal: pairs of [[0, a], [a, 0]]
  n = 8
  A = zeros(n, n)
  for i in 1:2:n
    v = randn()
    A[i, i+1] = v
    A[i+1, i] = v
  end

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test length(result.pivot_indices) == n
  @test length(result.neg_indices) == n ÷ 2  # half are negative

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── Real symmetric indefinite with strong off-diagonal dominance ──

@testitem "BK: strong off-diagonal dominance" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(903)

  # Matrix where off-diagonal elements are much larger than diagonal,
  # ensuring BK 2×2 path is taken (d < (1-5τ)*g condition)
  n = 10
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  # Large positive and negative eigenvalues
  λ = [100.0, -100.0, 50.0, -50.0, 10.0, -10.0, 1.0, -1.0, 0.1, -0.1]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test length(result.pivot_indices) == n
  @test length(result.neg_indices) == 5  # 5 negative eigenvalues

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── Verify neg_indices correctness ──

@testitem "BK: neg_indices matches eigenvalue signs" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(904)

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [5.0, 3.0, 1.0, 0.5, -0.3, -1.0, -2.0, -4.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-12, symmetry=:symmetric)
  @test length(result.pivot_indices) == n

  # Verify reconstruction using L * D_± * L'
  L = result.left
  k = size(L, 2)
  S = ones(k)
  for i in result.neg_indices
    S[i] = -1.0
  end
  A_approx = L * Diagonal(S) * L'
  @test norm(A - A_approx) / norm(A) < 1e-6

  # The eigenvalues of reconstructed matrix should match original
  evals_orig = sort(eigvals(Symmetric(A)))
  evals_approx = sort(eigvals(Symmetric(A_approx)))
  @test norm(evals_orig - evals_approx) / norm(evals_orig) < 1e-6

  # Number of negative indices should match number of negative eigenvalues
  @test length(result.neg_indices) == count(x -> x < 0, λ)
end

# ── Complex Hermitian indefinite (2×2 via eigen(Hermitian(B))) ──

@testitem "BK: complex hermitian indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(905)

  n = 8
  Q_raw = randn(ComplexF64, n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [4.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -4.0]
  A = Q * Diagonal(ComplexF64.(λ)) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  @test result.symmetry == :hermitian
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: complex hermitian anti-symmetric block" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  # 2×2 block with purely imaginary off-diagonals, zero real diagonal
  # This is Hermitian with eigenvalues ±|a|
  a = 3.0im
  A = ComplexF64[0.0   a;
                 conj(a) 0.0]

  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  @test length(result.pivot_indices) == 2
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "BK: complex hermitian zero-diagonal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(906)

  # Build Hermitian matrix with zero diagonal
  n = 6
  A = zeros(ComplexF64, n, n)
  for i in 1:n, j in i+1:n
    v = randn(ComplexF64)
    A[i, j] = v
    A[j, i] = conj(v)
  end

  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

# ── Complex symmetric indefinite (2×2 via Takagi factorization) ──

@testitem "BK: complex symmetric indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(907)

  n = 6
  # Complex symmetric: A = A^T (not A^H)
  V1 = randn(ComplexF64, n, 2)
  V2 = randn(ComplexF64, n, 2)
  A = V1 * transpose(V1) - 0.5 * V2 * transpose(V2)
  A = 0.5 * (A + transpose(A))

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "BK: complex symmetric zero diagonal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(908)

  # Complex symmetric with zero diagonal → forces 2×2
  n = 4
  A = zeros(ComplexF64, n, n)
  for i in 1:n, j in i+1:n
    v = randn(ComplexF64)
    A[i, j] = v
    A[j, i] = v  # symmetric, not hermitian
  end

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

# ── 2×2 fallback to 1×1 when one eigenvalue is below tolerance ──

@testitem "BK: 2x2 with one small eigenvalue" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  # Build a matrix where the 2×2 block has eigenvalues [large, tiny]
  # → 2×2 stores only one pivot, the other falls below tol
  n = 4
  λ = [10.0, 1.0, 1e-14, -10.0]
  A = Diagonal(λ)
  # Rotate to mix off-diagonal structure
  θ = π/6
  R = [cos(θ) -sin(θ) 0 0;
       sin(θ)  cos(θ) 0 0;
       0 0 cos(θ) -sin(θ);
       0 0 sin(θ)  cos(θ)]
  A = R * A * R'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  # Should find 3 significant pivots (eigenvalue ~1e-14 below tol)
  @test length(result.pivot_indices) == 3

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── lpaca with indefinite matrices ──

@testitem "BK: lpaca indefinite symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(909)

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = lpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: lpaca complex hermitian indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(910)

  n = 6
  Q_raw = randn(ComplexF64, n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = ComplexF64.([2.0, 1.0, 0.5, -0.5, -1.0, -2.0])
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = lpaca(A; tol=1e-10, symmetry=:hermitian)
  @test result.symmetry == :hermitian
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── lpaca complex symmetric (no Nyström – uses raw L-factors from Takagi) ──

@testitem "BK: lpaca complex symmetric zero diagonal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(930)

  # Zero diagonal forces 2×2 BK pivoting; lpaca (no Nyström) directly
  # validates the Takagi rotation (conj(U) in _eigendecompose_2x2).
  n = 6
  A = zeros(ComplexF64, n, n)
  for i in 1:n, j in i+1:n
    v = randn(ComplexF64)
    A[i, j] = v
    A[j, i] = v  # symmetric, not hermitian
  end

  result = lpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "BK: lpaca complex symmetric indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(931)

  n = 8
  V1 = randn(ComplexF64, n, 3)
  V2 = randn(ComplexF64, n, 3)
  A = V1 * transpose(V1) - 0.5 * V2 * transpose(V2)
  A = 0.5 * (A + transpose(A))

  result = lpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

# ── qrdalpaca with indefinite matrices ──

@testitem "BK: qrdalpaca indefinite symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(911)

  n = 10
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [5.0, 3.0, 1.0, 0.5, 0.1, -0.1, -0.5, -1.0, -3.0, -5.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = qrdalpaca(A; tol=1e-10, symmetry=:symmetric)
  @test result.symmetry == :symmetric
  @test length(result.neg_indices) == 5

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: qrdalpaca complex hermitian indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(912)

  n = 8
  Q_raw = randn(ComplexF64, n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = ComplexF64.([4.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -4.0])
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = qrdalpaca(A; tol=1e-10, symmetry=:hermitian)
  @test result.symmetry == :hermitian
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── SVD/Eigen extraction from indefinite matrices ──

@testitem "BK: alpaca_svd indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(913)

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [3.0, 2.0, 1.0, -0.5, -1.5, -2.0, -3.0, -4.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  U, S, Vt = alpaca_svd(Symmetric(A); tol=1e-10)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: alpaca_eigen indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(914)

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [3.0, 2.0, 1.0, -0.5, -1.5, -2.0, -3.0, -4.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  vals, vecs = alpaca_eigen(Symmetric(A); tol=1e-10)
  @test length(vals) == n
  # Eigenvalues should match original
  @test norm(sort(real.(vals)) - sort(λ)) / norm(λ) < 1e-6
end

@testitem "BK: alpaca_eigen complex hermitian indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(915)

  n = 6
  Q_raw = randn(ComplexF64, n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ_real = [3.0, 1.0, 0.5, -0.5, -1.0, -3.0]
  A = Q * Diagonal(ComplexF64.(λ_real)) * Q'
  A = 0.5 * (A + A')

  vals, vecs = alpaca_eigen(Hermitian(A); tol=1e-10)
  @test length(vals) == n
  @test norm(sort(real.(vals)) - sort(λ_real)) / norm(λ_real) < 1e-6
end

# ── Matrix-free interface with indefinite matrices ──

@testitem "BK: matrix-free symmetric indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(916)

  struct IndefiniteTestMatrix <: AbstractALPACAMatrix{Float64}
    data::Matrix{Float64}
  end
  Base.size(o::IndefiniteTestMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::IndefiniteTestMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j))
    return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::IndefiniteTestMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs)
      buf[k] = o.data[i, j]
    end
    return buf
  end

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [3.0, 1.0, 0.5, -0.5, -1.0, -3.0, -5.0, -7.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  mat = IndefiniteTestMatrix(A)
  opts = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(mat; options=opts)
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: matrix-free zero-diagonal symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(917)

  struct ZeroDiagSymTestMatrix <: AbstractALPACAMatrix{Float64}
    data::Matrix{Float64}
  end
  Base.size(o::ZeroDiagSymTestMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::ZeroDiagSymTestMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j))
    return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::ZeroDiagSymTestMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs)
      buf[k] = o.data[i, j]
    end
    return buf
  end

  # Block-diagonal symmetric with zero diagonal
  n = 6
  A = zeros(n, n)
  for i in 1:2:n
    v = 2.0 + randn()^2
    A[i, i+1] = v
    A[i+1, i] = v
  end

  mat = ZeroDiagSymTestMatrix(A)
  opts = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(mat; options=opts)
  @test length(result.pivot_indices) == n
  @test length(result.neg_indices) == n ÷ 2

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── Takagi decomposition for indefinite complex symmetric ──

@testitem "BK: alpaca_takagi indefinite complex symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(918)

  n = 6
  V1 = randn(ComplexF64, n, 2)
  V2 = randn(ComplexF64, n, 2)
  A = V1 * transpose(V1) - 0.3 * V2 * transpose(V2)
  A = 0.5 * (A + transpose(A))

  U, D = alpaca_takagi(A; tol=1e-10, symmetry=:symmetric)
  @test all(D .>= 0)
  A_approx = U * Diagonal(D) * transpose(U)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

# ── QR extraction from indefinite ──

@testitem "BK: alpaca_qr indefinite symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(919)

  n = 8
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = [3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0]
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  Qd, R = alpaca_qr(Symmetric(A); tol=1e-10)
  @test norm(Qd'Qd - I(size(Qd, 2))) < 1e-10
  A_approx = Qd * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ── Near-singular with mixed signs near tolerance boundary ──

@testitem "BK: eigenvalues near tolerance boundary" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(920)

  n = 6
  tol = 1e-8
  # Eigenvalues: two above tol, two below tol (should be dropped), two negative
  λ = [1.0, -1.0, tol * 0.5, -tol * 0.5, tol * 0.1, -tol * 0.1]
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=tol, symmetry=:symmetric)
  # Should find ~2 significant pivots (λ=±1), plus possibly borderline
  # pivots near tol that are captured by smooth scaling
  @test length(result.pivot_indices) <= 4

  A_approx = reconstruct(result)
  # Error should be on the order of the dropped eigenvalues
  @test norm(A - A_approx) < tol * 10
end

# ── Large indefinite matrix (ensures scaling works) ──

@testitem "BK: large indefinite matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(921)

  n = 50
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  # 25 positive, 25 negative eigenvalues
  λ = vcat(range(1.0, 10.0, length=25), range(-10.0, -1.0, length=25))
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test length(result.pivot_indices) == n
  @test length(result.neg_indices) == 25

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "BK: large complex hermitian indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(922)

  n = 30
  Q_raw = randn(ComplexF64, n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  λ = ComplexF64.(vcat(range(1.0, 5.0, length=15), range(-5.0, -1.0, length=15)))
  A = Q * Diagonal(λ) * Q'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  @test length(result.pivot_indices) == n
  @test length(result.neg_indices) == 15

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end
