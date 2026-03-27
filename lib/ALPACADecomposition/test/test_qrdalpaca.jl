@testitem "qrdalpaca: real symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(301)
  n = 10
  V = randn(n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  res = qrdalpaca(Symmetric(A); tol=1e-10)
  @test res.symmetry == :symmetric
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: complex hermitian" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(302)
  n = 8
  V = randn(ComplexF64, n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  res = qrdalpaca(Hermitian(A); tol=1e-10)
  @test res.symmetry == :hermitian
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(303)
  n = 8
  A = randn(n, n)

  res = qrdalpaca(A; tol=1e-10)
  @test res.symmetry == :general
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: rectangular" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(304)
  A = randn(10, 6)

  res = qrdalpaca(A; tol=1e-10)
  @test res.symmetry == :general
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: options keyword" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(305)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  opts = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  res = qrdalpaca(A; options=opts)
  @test res.symmetry == :symmetric
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: complex symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(306)
  n = 8
  V = randn(ComplexF64, n, 3)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  res = qrdalpaca(A; tol=1e-10, symmetry=:symmetric)
  @test res.symmetry == :symmetric
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: matrix-free interface" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(307)
  n = 6
  V = randn(n, 2)
  A = V * V'
  A = 0.5 * (A + A')

  mat = DenseALPACAMatrix(A)
  options = ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true)
  res = qrdalpaca(mat; options=options)
  @test norm(A - reconstruct(res)) / norm(A) < 1e-6
end
