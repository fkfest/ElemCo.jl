@testitem "Cache construction: symmetric" begin
  using ALPACADecomposition

  pairs = [(1,1), (2,2), (3,3)]
  cache = ALPACACache(Float64, 3, :symmetric, pairs)
  @test cache.n_cols == 0
  @test cache.n_rows == 0
  @test size(cache.rows) == (0, 0)  # no row cache for symmetric
  @test length(cache.cbuf) == 3
  @test length(cache.rbuf) == 3
  @test length(cache.principal_values) == 3
  @test !cache.principal_fetched
end

@testitem "Cache construction: general" begin
  using ALPACADecomposition

  cache = ALPACACache(Float64, 4, :general, [(1,2), (3,4)])
  @test size(cache.rows, 1) == 4
  @test size(cache.rows, 2) > 0
  @test length(cache.row_map) == 4
end

@testitem "Cache construction: complex" begin
  using ALPACADecomposition

  pairs = [(1,1), (2,2), (3,3)]
  cache = ALPACACache(ComplexF64, 3, :symmetric, pairs)
  @test eltype(cache.cbuf) == ComplexF64
  @test eltype(cache.columns) == ComplexF64
end

@testitem "Cache column storage" begin
  using ALPACADecomposition

  pairs = [(1,1), (2,2), (3,3)]
  cache = ALPACACache(Float64, 3, :symmetric, pairs)

  col1 = [1.0, 2.0, 3.0]
  col2 = [4.0, 5.0, 6.0]

  k1 = ALPACADecomposition.store_column!(cache, 2, col1)
  @test k1 == 1
  @test cache.n_cols == 1
  @test cache.col_map[2] == 1
  @test cache.col_index[1] == 2
  @test cache.columns[:, 1] == col1

  k2 = ALPACADecomposition.store_column!(cache, 3, col2)
  @test k2 == 2
  @test cache.n_cols == 2
  @test cache.col_map[3] == 2
end

@testitem "Cache column amortized doubling" begin
  using ALPACADecomposition

  cache_big = ALPACACache(Float64, 5, :symmetric, [(i,i) for i in 1:5])
  for j in 1:5
    ALPACADecomposition.store_column!(cache_big, j, Float64.(collect(1:5) .* j))
  end
  @test cache_big.n_cols == 5
  @test cache_big.columns[:, 3] == [3.0, 6.0, 9.0, 12.0, 15.0]
end

@testitem "Cache row storage (general)" begin
  using ALPACADecomposition

  pairs = [(1,2)]
  cache = ALPACACache(Float64, 3, :general, pairs)

  row1 = [10.0, 20.0, 30.0]
  k = ALPACADecomposition.store_row!(cache, 1, row1)
  @test k == 1
  @test cache.n_rows == 1
  @test cache.row_map[1] == 1
  @test cache.rows[:, 1] == row1
end

@testitem "Cache pivot tracking" begin
  using ALPACADecomposition

  pairs = [(1,1), (2,2), (3,3)]
  cache = ALPACACache(Float64, 3, :symmetric, pairs)

  ALPACADecomposition.store_pivot!(cache, 2, 5.0)
  @test cache.pivot_indices == [2]
  @test cache.is_pivot[2] == true
  @test cache.is_pivot[1] == false
  @test cache.pivot_diag[1] == 5.0

  ALPACADecomposition.store_pivot!(cache, 1, 3.0)
  @test cache.pivot_indices == [2, 1]
  @test cache.pivot_diag[2] == 3.0
end

@testitem "Cache principal init: PrincipalPairs" begin
  using ALPACADecomposition

  matrix = [1.0 2.0 3.0; 2.0 5.0 6.0; 3.0 6.0 9.0]
  mat = DenseALPACAMatrix(matrix)
  pairs = [(1,1), (2,2), (3,3)]

  cache = ALPACACache(Float64, 3, :symmetric, pairs)
  desc = PrincipalPairs(pairs)
  ALPACADecomposition.init_principal_values!(cache, mat, desc)
  @test cache.principal_fetched == true
  @test cache.principal_values == [1.0, 5.0, 9.0]

  # Second call is a no-op
  cache.principal_values[1] = 999.0
  ALPACADecomposition.init_principal_values!(cache, mat, desc)
  @test cache.principal_values[1] == 999.0  # not overwritten
end

@testitem "Cache principal init: PrincipalTriples" begin
  using ALPACADecomposition

  matrix = [1.0 2.0 3.0; 2.0 5.0 6.0; 3.0 6.0 9.0]
  mat = DenseALPACAMatrix(matrix)
  pairs = [(1,1), (2,2), (3,3)]

  cache = ALPACACache(Float64, 3, :symmetric, pairs)
  desc = PrincipalTriples(pairs, [10.0, 20.0, 30.0])
  ALPACADecomposition.init_principal_values!(cache, mat, desc)
  @test cache.principal_fetched == true
  @test cache.principal_values == [10.0, 20.0, 30.0]
end
