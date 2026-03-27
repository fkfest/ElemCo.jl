# ──────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────

@testitem "Error: hermitian rejects rectangular" begin
  using ALPACADecomposition
  using LinearAlgebra, Random

  A = randn(ComplexF64, 4, 6)
  options = ALPACAOptions(tol=1e-10, symmetry=:hermitian)
  @test_throws ArgumentError alpaca(A; options=options)
end

@testitem "Error: negative dimension in descriptor" begin
  using ALPACADecomposition

  @test_throws ArgumentError normalize_principal_descriptor(:symmetric, -1, nothing)
end

@testitem "Error: takagi on non-symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(501)
  n = 6
  A = randn(n, n)

  # General matrix → takagi should error
  @test_throws Exception alpaca_takagi(A; tol=1e-10)
end

@testitem "Error: lpaca without tol or options" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(502)
  S = randn(4, 4); S = S + S'
  @test_throws ArgumentError lpaca(S)
end

@testitem "Error: qrdalpaca without tol or options" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(503)
  S = randn(4, 4); S = S + S'
  @test_throws ArgumentError qrdalpaca(S)
end

# ──────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────

@testitem "Edge: 1×1 matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  A = fill(5.0, 1, 1)
  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :symmetric
  @test length(result.pivot_indices) == 1
  @test norm(A - reconstruct(result)) < 1e-10
end

@testitem "Edge: 1×1 complex hermitian" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  A = fill(3.0 + 0.0im, 1, 1)
  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  @test result.symmetry == :hermitian
  @test norm(A - reconstruct(result)) < 1e-10
end

@testitem "Edge: 1×1 general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  A = fill(7.0, 1, 1)
  result = alpaca(A; tol=1e-10, symmetry=:general)
  @test result.symmetry == :general
  @test norm(A - reconstruct(result)) < 1e-10
end

@testitem "Edge: large tolerance → few pivots" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(510)
  n = 10
  # Low-rank matrix: only 2 significant pivots
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test length(result.pivot_indices) <= 3  # rank ~2, should need few pivots
end

@testitem "Edge: identity matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  n = 5
  A = Matrix(1.0I, n, n)
  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :symmetric
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Edge: diagonal matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra

  A = diagm([4.0, 3.0, 2.0, 1.0, 0.5])
  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :symmetric
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Edge: rank-deficient general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(511)
  m, n = 10, 8
  r = 2
  U = randn(m, r)
  V = randn(n, r)
  A = U * V'

  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :general
  @test length(result.pivot_indices) <= r + 2  # should find ~r pivots
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Edge: complex symmetric zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(ComplexF64, 4, 4)
  result = alpaca(A; tol=1e-10, symmetry=:symmetric)
  @test length(result.pivot_indices) == 0
  @test size(result.left) == (4, 0)
end

@testitem "Edge: hermitian zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(ComplexF64, 4, 4)
  result = alpaca(A; tol=1e-10, symmetry=:hermitian)
  @test length(result.pivot_indices) == 0
  @test size(result.left) == (4, 0)
end

@testitem "Edge: general zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  result = alpaca(A; tol=1e-10, symmetry=:general)
  @test length(result.pivot_indices) == 0
  @test size(result.left) == (4, 0)
end

@testitem "Edge: explicit symmetry override" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(512)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  # Force general treatment of a symmetric matrix
  result = alpaca(A; tol=1e-10, symmetry=:general)
  @test result.symmetry == :general
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Edge: max_rank = 1" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(513)
  n = 6
  A_raw = randn(n, n)
  A = A_raw' * A_raw + I

  result = alpaca(A; tol=1e-14, symmetry=:symmetric, max_rank=1)
  @test length(result.pivot_indices) == 1
end

@testitem "Edge: SVD extraction from zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  U, S, Vt = alpaca_svd(A; tol=1e-10)
  @test isempty(S)
  @test size(U) == (4, 0)
  @test size(Vt) == (0, 4)
end

@testitem "Edge: eigen extraction from zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  vals, vecs = alpaca_eigen(A; tol=1e-10)
  @test isempty(vals)
  @test size(vecs) == (4, 0)
end

@testitem "Edge: QR extraction from zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  Q, R = alpaca_qr(A; tol=1e-10)
  @test size(Q) == (4, 0)
  @test size(R) == (0, 4)
end
