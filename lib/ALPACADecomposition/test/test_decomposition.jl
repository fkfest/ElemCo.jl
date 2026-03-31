@testitem "alpaca: real symmetric PSD" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(101)
  n = 8
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test result.symmetry == :symmetric
  @test length(result.pivot_indices) >= 2
  @test length(result.pivot_indices) <= n
  @test isempty(result.row_pivots)

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: real symmetric full rank" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(102)
  n = 6
  A_raw = randn(n, n)
  A = A_raw' * A_raw + I

  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test result.symmetry == :symmetric
  @test length(result.pivot_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca: real symmetric indefinite" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(103)
  n = 6
  Q_raw = randn(n, n)
  Q, _ = qr(Q_raw)
  Q = Matrix(Q)
  lambda = [3.0, 2.0, 1.0, -0.5, -1.5, -2.0]
  A = Q * Diagonal(lambda) * Q'
  A = 0.5 * (A + A')

  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test result.symmetry == :symmetric
  @test length(result.neg_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca: complex hermitian" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(104)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  options = ALPACAOptions(tol=1e-10, symmetry=:hermitian)
  result = alpaca(A; options=options)

  @test result.symmetry == :hermitian
  @test length(result.pivot_indices) >= 3

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: complex symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(105)
  n = 6
  V = randn(ComplexF64, n, 2)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test result.symmetry == :symmetric

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: general matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(106)
  n = 6
  U = randn(n, 3)
  V = randn(n, 3)
  A = U * V'

  principal = [(i, i) for i in 1:n]
  options = ALPACAOptions(tol=1e-10, symmetry=:general)
  result = alpaca(A; principal=principal, options=options)

  @test result.symmetry == :general
  @test length(result.pivot_indices) >= 1
  @test length(result.row_pivots) >= 1
  @test length(result.row_pivots) == length(result.pivot_indices)

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: general weakly-coupled blocks" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  # Test that ACA does not stop prematurely when the last row's
  # deflated values underestimate the remaining residual due to
  # weak cross-block coupling.
  Random.seed!(42)
  m, n = 80, 70; r2 = 15
  B1 = randn(m, 20)
  B1[31:80, :] .= 1e-9 * randn(50, 20)
  V1 = randn(20, 20)
  U2 = zeros(m, r2)
  U2[31:80, :] = randn(50, r2)
  U2[1:30, :] = 1e-9 * randn(30, r2)
  V2 = randn(50, r2)
  A = zeros(m, n)
  A[:, 1:20] = B1 * V1
  A[:, 21:70] = U2 * V2'

  principal = PrincipalPairs([(1, 1)])
  for tol in [1e-4, 1e-6]
    result = alpaca(A; tol=tol, symmetry=:general, principal=principal)
    A_approx = reconstruct(result)
    rk = size(result.left, 2)
    @test rk == 35
    @test norm(A - A_approx) / norm(A) < 1e-10
  end
end

@testitem "alpaca: zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  n = 4
  A = zeros(n, n)
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test length(result.pivot_indices) == 0
  @test size(result.left) == (n, 0)
end

@testitem "alpaca: rank-1 matrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(108)
  n = 5
  v = randn(n)
  A = v * v'

  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; options=options)

  @test length(result.pivot_indices) >= 1

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: max_rank truncation" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(109)
  n = 8
  A_raw = randn(n, n)
  A = A_raw' * A_raw + I

  options = ALPACAOptions(tol=1e-14, symmetry=:symmetric, max_rank=3)
  result = alpaca(A; options=options)

  @test length(result.pivot_indices) <= 3
end

@testitem "alpaca: with PrincipalTriples" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(110)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  diag_vals = [(i, i, A[i,i]) for i in 1:n]
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(A; principal=diag_vals, options=options)

  @test length(result.pivot_indices) >= 2

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: DenseALPACAMatrix" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(111)
  n = 5
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  mat = DenseALPACAMatrix(A)
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(mat; options=options)

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: rectangular tall" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(112)
  m, n = 8, 5
  r = 3
  U = randn(m, r)
  V = randn(n, r)
  A = U * V'

  principal = [(i, i) for i in 1:min(m, n)]
  options = ALPACAOptions(tol=1e-10, symmetry=:general)
  result = alpaca(A; principal=principal, options=options)

  @test result.symmetry == :general
  @test length(result.pivot_indices) >= 1
  @test size(result.left, 1) == m
  @test size(result.right, 1) == n

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: rectangular wide" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(113)
  m, n = 5, 10
  r = 2
  U = randn(m, r)
  V = randn(n, r)
  A = U * V'

  principal = [(i, i) for i in 1:min(m, n)]
  options = ALPACAOptions(tol=1e-10, symmetry=:general)
  result = alpaca(A; principal=principal, options=options)

  @test result.symmetry == :general
  @test size(result.left, 1) == m
  @test size(result.right, 1) == n

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "alpaca: symmetric rejects rectangular" begin
  using ALPACADecomposition
  using LinearAlgebra, Random

  A = randn(4, 6)
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  @test_throws ArgumentError alpaca(A; options=options)
end

@testitem "lpaca: real symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  result = lpaca(Symmetric(A); tol=1e-10)
  @test result.symmetry == :symmetric

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "lpaca: complex hermitian" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(202)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  result = lpaca(Hermitian(A); tol=1e-10)
  @test result.symmetry == :hermitian

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "lpaca: general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(203)
  n = 6
  A = randn(n, n)

  result = lpaca(A; tol=1e-10)
  @test result.symmetry == :general
  @test !isempty(result.row_pivots)

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end

@testitem "lpaca: matrix-free interface" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(204)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  mat = DenseALPACAMatrix(A)
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = lpaca(mat; options=options)

  A_approx = reconstruct(result)
  @test norm(A - A_approx) / max(norm(A), 1.0) < 1e-6
end
