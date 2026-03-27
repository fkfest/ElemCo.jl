@testitem "Auto-detect: real symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  S = randn(n, n); S = S + S'
  res = alpaca(S; tol=1e-10)
  @test res.symmetry == :symmetric
  @test norm(S - reconstruct(res)) / norm(S) < 1e-6
end

@testitem "Auto-detect: complex hermitian" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  H = randn(ComplexF64, n, n); H = H + H'
  res = alpaca(H; tol=1e-10)
  @test res.symmetry == :hermitian
  @test norm(H - reconstruct(res)) / norm(H) < 1e-6
end

@testitem "Auto-detect: complex symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  CS = randn(ComplexF64, n, n); CS = CS + transpose(CS)
  res = alpaca(CS; tol=1e-10)
  @test res.symmetry == :symmetric
  @test norm(CS - reconstruct(res)) / norm(CS) < 1e-6
end

@testitem "Auto-detect: general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  G = randn(n, n)
  res = alpaca(G; tol=1e-10)
  @test res.symmetry == :general
  @test norm(G - reconstruct(res)) / norm(G) < 1e-6
end

@testitem "Auto-detect: rectangular → general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  R = randn(8, 5)
  res = alpaca(R; tol=1e-10)
  @test res.symmetry == :general
  @test norm(R - reconstruct(res)) / norm(R) < 1e-6
end

@testitem "Convenience: error without tol or options" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(201)
  n = 6
  S = randn(n, n); S = S + S'
  @test_throws ArgumentError alpaca(S)
end
