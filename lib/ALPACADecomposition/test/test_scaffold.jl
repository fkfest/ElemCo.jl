@testitem "Matrix-free interface" begin
  using ALPACADecomposition
  using LinearAlgebra

  matrix = [1.0 2.0 3.0; 2.0 5.0 6.0; 3.0 6.0 9.0]
  mat = DenseALPACAMatrix(matrix)

  column_buffer = zeros(3)
  row_buffer = zeros(3)
  elem_buffer = zeros(2)

  @test size(mat) == (3, 3)
  @test column!(column_buffer, mat, 2) == [2.0, 5.0, 6.0]
  @test row!(row_buffer, mat, 2) == [2.0, 5.0, 6.0]
  @test elements!(elem_buffer, mat, [(1, 1), (2, 3)]) == [1.0, 6.0]
end

@testitem "Principal descriptor normalization" begin
  using ALPACADecomposition

  descriptor_default = normalize_principal_descriptor(:symmetric, 3, nothing)
  @test descriptor_default isa PrincipalPairs
  @test descriptor_default.pairs == [(1, 1), (2, 2), (3, 3)]

  descriptor_pairs = normalize_principal_descriptor(:general, 3, [(1, 2), (2, 3)])
  @test descriptor_pairs isa PrincipalPairs
  @test descriptor_pairs.pairs == [(1, 2), (2, 3)]

  descriptor_triples = normalize_principal_descriptor(:general, 3,
    [(1, 2, 0.25), (3, 1, -0.5)])
  @test descriptor_triples isa PrincipalTriples
  @test descriptor_triples.pairs == [(1, 2), (3, 1)]
  @test descriptor_triples.values == [0.25, -0.5]

  # Pass-through for existing descriptors
  pp = PrincipalPairs([(1,1)])
  @test normalize_principal_descriptor(:symmetric, 3, pp) === pp

  pt = PrincipalTriples([(1,1)], [1.0])
  @test normalize_principal_descriptor(:symmetric, 3, pt) === pt
end

@testitem "ALPACAOptions construction" begin
  using ALPACADecomposition

  options = ALPACAOptions(tol=1e-8, symmetry=:hermitian)
  @test options.tol == 1e-8
  @test options.pivotol == 1e-8
  @test options.qr == false
  @test options.symmetry == :hermitian
  @test options.sigma == 0.01
  @test options.max_rank == typemax(Int)

  # Custom pivotol
  options2 = ALPACAOptions(tol=1e-8, symmetry=:symmetric, pivotol=1e-6)
  @test options2.pivotol == 1e-6

  # Copy constructor
  options3 = ALPACAOptions(options; qr=true, max_rank=5)
  @test options3.tol == 1e-8
  @test options3.qr == true
  @test options3.max_rank == 5
end
