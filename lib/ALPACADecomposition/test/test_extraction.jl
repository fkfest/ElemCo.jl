# ──────────────────────────────────────────────────────────────────
# SVD extraction
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_svd: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(401)
  n = 8
  V = randn(n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  U, S, Vt = alpaca_svd(Symmetric(A); tol=1e-10)
  @test length(S) >= 3
  @test all(S .>= 0)
  @test issorted(S; rev=true)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_svd: complex hermitian" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(402)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  U, S, Vt = alpaca_svd(Hermitian(A); tol=1e-10)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_svd: general" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(403)
  n = 6
  A = randn(n, n)

  U, S, Vt = alpaca_svd(A; tol=1e-10)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_svd: rectangular" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(404)
  A = randn(8, 5)

  U, S, Vt = alpaca_svd(A; tol=1e-10)
  @test size(U, 1) == 8
  @test size(Vt, 2) == 5
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "lpaca_svd: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(405)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  U, S, Vt = lpaca_svd(Symmetric(A); tol=1e-10)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "qrdalpaca_svd: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(406)
  n = 8
  V = randn(n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  U, S, Vt = qrdalpaca_svd(Symmetric(A); tol=1e-10)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Eigen extraction
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_eigen: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(411)
  n = 6
  A_raw = randn(n, n)
  A = A_raw' * A_raw + I

  vals, vecs = alpaca_eigen(Symmetric(A); tol=1e-10)
  @test length(vals) >= 1
  # Verify A * v ≈ λ * v
  for i in 1:length(vals)
    @test norm(A * vecs[:, i] - vals[i] * vecs[:, i]) / abs(vals[i]) < 1e-4
  end
end

@testitem "alpaca_eigen: real symmetric indefinite" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(412)
  n = 6
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  lambda = [3.0, 2.0, 1.0, -0.5, -1.5, -2.0]
  A = Q * Diagonal(lambda) * Q'
  A = 0.5 * (A + A')

  vals, vecs = alpaca_eigen(Symmetric(A); tol=1e-10)
  # Should have both positive and negative eigenvalues
  @test any(vals .> 0)
  @test any(vals .< 0)
  # Reconstruction
  A_approx = vecs * Diagonal(vals) * vecs'
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_eigen: complex hermitian" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(413)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  vals, vecs = alpaca_eigen(Hermitian(A); tol=1e-10)
  @test all(v -> v isa Real, vals)
  A_approx = vecs * Diagonal(vals) * vecs'
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_eigen: general" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(414)
  n = 6
  A = randn(n, n)

  vals, vecs = alpaca_eigen(A; tol=1e-10)
  # For general, reconstruction is A ≈ vecs * diag(vals) * inv(vecs)
  A_approx = vecs * Diagonal(vals) * inv(vecs)
  @test norm(A - A_approx) / norm(A) < 1e-4
end

@testitem "lpaca_eigen: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(415)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  vals, vecs = lpaca_eigen(Symmetric(A); tol=1e-10)
  A_approx = vecs * Diagonal(vals) * vecs'
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "qrdalpaca_eigen: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(416)
  n = 8
  V = randn(n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  vals, vecs = qrdalpaca_eigen(Symmetric(A); tol=1e-10)
  A_approx = vecs * Diagonal(vals) * vecs'
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Takagi extraction (complex symmetric only)
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_takagi: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(421)
  n = 6
  V = randn(ComplexF64, n, 2)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  U, D = alpaca_takagi(A; tol=1e-10, symmetry=:symmetric)
  @test all(D .>= 0)
  # Takagi: A = U * diag(D) * U^T
  A_approx = U * Diagonal(D) * transpose(U)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "lpaca_takagi: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(422)
  n = 6
  V = randn(ComplexF64, n, 2)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  U, D = lpaca_takagi(A; tol=1e-10, symmetry=:symmetric)
  @test all(D .>= 0)
  A_approx = U * Diagonal(D) * transpose(U)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "qrdalpaca_takagi: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(423)
  n = 6
  V = randn(ComplexF64, n, 2)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  U, D = qrdalpaca_takagi(A; tol=1e-10, symmetry=:symmetric)
  @test all(D .>= 0)
  A_approx = U * Diagonal(D) * transpose(U)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# QR extraction
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_qr: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(431)
  n = 6
  V = randn(n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  Q, R = alpaca_qr(Symmetric(A); tol=1e-10)
  # Q should have orthonormal columns
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_qr: general" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(432)
  A = randn(6, 6)

  Q, R = alpaca_qr(A; tol=1e-10)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "lpaca_qr: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(433)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  Q, R = lpaca_qr(Symmetric(A); tol=1e-10)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "qrdalpaca_qr: real symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(434)
  n = 8
  V = randn(n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  Q, R = qrdalpaca_qr(Symmetric(A); tol=1e-10)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end
