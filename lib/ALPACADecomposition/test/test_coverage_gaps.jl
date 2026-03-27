# ──────────────────────────────────────────────────────────────────
# ALPACAOptions validation
# ──────────────────────────────────────────────────────────────────

@testitem "Options: negative pivotol" begin
  using ALPACADecomposition
  @test_throws ArgumentError ALPACAOptions(tol=1e-10, pivotol=-1.0)
end

@testitem "Options: negative tol" begin
  using ALPACADecomposition
  @test_throws ArgumentError ALPACAOptions(tol=-1e-10)
end

@testitem "Options: negative sigma" begin
  using ALPACADecomposition
  @test_throws ArgumentError ALPACAOptions(tol=1e-10, sigma=-0.5)
end

@testitem "Options: negative max_rank" begin
  using ALPACADecomposition
  @test_throws ArgumentError ALPACAOptions(tol=1e-10, max_rank=-1)
end

@testitem "Options: invalid symmetry" begin
  using ALPACADecomposition
  @test_throws ArgumentError ALPACAOptions(tol=1e-10, symmetry=:invalid)
end

@testitem "Options: copy-with-modification" begin
  using ALPACADecomposition
  opts = ALPACAOptions(tol=1e-8)
  opts2 = ALPACAOptions(opts; tol=1e-12, symmetry=:general)
  @test opts2.tol == 1e-12
  @test opts2.symmetry == :general
  @test opts2.sigma == opts.sigma
end

# ──────────────────────────────────────────────────────────────────
# PrincipalPairs descriptor path
# ──────────────────────────────────────────────────────────────────

@testitem "Descriptors: principal_triples with pairs" begin
  using ALPACADecomposition
  pairs = [(1, 2), (3, 4)]
  desc = principal_triples(pairs)
  @test desc isa PrincipalPairs
end

# ──────────────────────────────────────────────────────────────────
# Custom matrix-free (exercises AbstractALPACAMatrix code paths)
# ──────────────────────────────────────────────────────────────────

@testitem "Custom matrix-free: symmetric via column!/elements!" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(601)

  # Build a custom mat that wraps a dense matrix but isn't DenseALPACAMatrix
  struct TestMatrix <: AbstractALPACAMatrix
    data::Matrix{Float64}
  end
  Base.size(o::TestMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::TestMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j))
    return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::TestMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs)
      buf[k] = o.data[i, j]
    end
    return buf
  end

  n = 8
  V = randn(n, 3)
  A = V * V' + 0.01I
  A = 0.5 * (A + A')

  mat = TestMatrix(A)
  opts = ALPACAOptions(tol=1e-10, symmetry=:symmetric)
  result = alpaca(mat; options=opts)
  @test result.symmetry == :symmetric
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Custom matrix-free: general via column!/row!/elements!" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(602)

  struct TestMatrixGeneral <: AbstractALPACAMatrix
    data::Matrix{Float64}
  end
  Base.size(o::TestMatrixGeneral) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::TestMatrixGeneral, j::Integer)
    copyto!(buf, view(o.data, :, j))
    return buf
  end
  function ALPACADecomposition.row!(buf::AbstractVector, o::TestMatrixGeneral, i::Integer)
    copyto!(buf, view(o.data, i, :))
    return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::TestMatrixGeneral,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs)
      buf[k] = o.data[i, j]
    end
    return buf
  end

  n = 8
  A = randn(n, n)

  mat = TestMatrixGeneral(A)
  opts = ALPACAOptions(tol=1e-10, symmetry=:general)
  result = alpaca(mat; options=opts)
  @test result.symmetry == :general
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Complex symmetric SVD/Eigen/QR extraction (uncovered decomp paths)
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_svd: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(611)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  U, S, Vt = alpaca_svd(A; tol=1e-10, symmetry=:symmetric)
  @test all(S .>= 0)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_eigen: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(612)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  vals, vecs = alpaca_eigen(A; tol=1e-10, symmetry=:symmetric)
  # Complex symmetric → general eigen path
  @test length(vals) >= 1
end

@testitem "alpaca_qr: complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(613)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  Q, R = alpaca_qr(A; tol=1e-10, symmetry=:symmetric)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_qr: general" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(614)
  n = 6
  A = randn(n, n)

  Q, R = alpaca_qr(A; tol=1e-10)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "alpaca_qr: hermitian" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(615)
  n = 6
  V = randn(ComplexF64, n, 3)
  A = V * V'
  A = 0.5 * (A + A')

  Q, R = alpaca_qr(A; tol=1e-10)
  @test norm(Q'Q - I(size(Q, 2))) < 1e-10
  A_approx = Q * R
  @test norm(A - A_approx) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Eigen error: non-square general
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca_eigen: rectangular general errors" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(621)
  A = randn(8, 5)

  @test_throws ArgumentError alpaca_eigen(A; tol=1e-10)
end

# ──────────────────────────────────────────────────────────────────
# Rank-0 extraction paths
# ──────────────────────────────────────────────────────────────────

@testitem "SVD extraction: rank-0 matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  U, S, Vt = alpaca_svd(A; tol=1e-10)
  @test length(S) == 0
  @test size(U) == (4, 0)
  @test size(Vt) == (0, 4)
end

@testitem "Eigen extraction: rank-0 matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  vals, vecs = alpaca_eigen(A; tol=1e-10)
  @test length(vals) == 0
  @test size(vecs) == (4, 0)
end

@testitem "QR extraction: rank-0 matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(4, 4)
  Q, R = alpaca_qr(A; tol=1e-10)
  @test size(Q) == (4, 0)
  @test size(R) == (0, 4)
end

@testitem "Takagi extraction: rank-0 complex symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(ComplexF64, 4, 4)
  U, D = alpaca_takagi(A; tol=1e-10, symmetry=:symmetric)
  @test length(D) == 0
  @test size(U) == (4, 0)
end

# ──────────────────────────────────────────────────────────────────
# qrdalpaca with matrix-free + QR refinement trigger
# ──────────────────────────────────────────────────────────────────

@testitem "qrdalpaca: general DenseALPACAMatrix triggers QR refinement" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(3)

  # Low-rank general matrix with small noise — pivotol >> noise causes ALPACA to
  # stop early while QR refinement discovers additional significant columns.
  n = 30; k = 5
  U = randn(n, k); V = randn(n, k)
  A = U * V' + 0.005 * randn(n, n)

  mat = DenseALPACAMatrix(A)
  opts_plain = ALPACAOptions(tol=1e-8, symmetry=:general, pivotol=0.01)
  opts_qrd   = ALPACAOptions(tol=1e-8, symmetry=:general, qr=true, pivotol=0.01)

  r_plain = alpaca(mat; options=opts_plain)
  r_qrd   = qrdalpaca(mat; options=opts_qrd)
  @test length(r_qrd.pivot_indices) > length(r_plain.pivot_indices)
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 0.01
end

@testitem "qrdalpaca: general custom mat triggers QR refinement" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(3)

  n = 30; k = 5
  U = randn(n, k); V = randn(n, k)
  A = U * V' + 0.005 * randn(n, n)

  struct QRDGenMatrix <: AbstractALPACAMatrix
    data::Matrix{Float64}
  end
  Base.size(o::QRDGenMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::QRDGenMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j)); return buf
  end
  function ALPACADecomposition.row!(buf::AbstractVector, o::QRDGenMatrix, i::Integer)
    copyto!(buf, view(o.data, i, :)); return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::QRDGenMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs); buf[k] = o.data[i, j]; end; return buf
  end

  mat = QRDGenMatrix(A)
  opts_plain = ALPACAOptions(tol=1e-8, symmetry=:general, pivotol=0.01)
  opts_qrd   = ALPACAOptions(tol=1e-8, symmetry=:general, qr=true, pivotol=0.01)

  r_plain = alpaca(mat; options=opts_plain)
  r_qrd   = qrdalpaca(mat; options=opts_qrd)
  @test length(r_qrd.pivot_indices) > length(r_plain.pivot_indices)
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 0.01
end

@testitem "qrdalpaca: complex symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(633)
  n = 10
  V = randn(ComplexF64, n, 4)
  A = V * transpose(V)
  A = 0.5 * (A + transpose(A))

  result = qrdalpaca(A; tol=1e-12, symmetry=:symmetric)
  @test result.symmetry == :symmetric
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Block diagonal + bad principal → symmetric QR refinement
# ──────────────────────────────────────────────────────────────────

@testitem "qrdalpaca: symmetric block diagonal with bad principal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  # Block diagonal: strong block 1, weak block 2.
  # Bad principal points only at block 1 → ALPACA misses block 2 entirely.
  # QR refinement discovers the missing block → _symmetric_refactorize path.
  n = 20
  B1 = randn(10, 10); B1 = B1' * B1 + 10I   # strong block, eigenvalues ~10
  B2 = randn(10, 10); B2 = 0.1 * (B2' * B2) + 0.1I  # weak block, eigenvalues ~0.1
  A = zeros(n, n)
  A[1:10, 1:10] .= B1
  A[11:20, 11:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  opts_plain = ALPACAOptions(tol=1e-8, symmetry=:symmetric)
  opts_qrd   = ALPACAOptions(tol=1e-8, symmetry=:symmetric, qr=true)

  mat = DenseALPACAMatrix(A)
  r_plain = alpaca(mat; options=opts_plain, principal=bad_principal)
  r_qrd   = qrdalpaca(mat; options=opts_qrd, principal=bad_principal)

  # Plain ALPACA only finds block 1; qrdalpaca finds both blocks
  @test length(r_qrd.pivot_indices) > length(r_plain.pivot_indices)
  @test length(r_qrd.pivot_indices) == n
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: symmetric matrix-free block diagonal with bad principal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  n = 20
  B1 = randn(10, 10); B1 = B1' * B1 + 10I
  B2 = randn(10, 10); B2 = 0.1 * (B2' * B2) + 0.1I
  A = zeros(n, n)
  A[1:10, 1:10] .= B1
  A[11:20, 11:20] .= B2

  struct BlockMatrix <: AbstractALPACAMatrix
    data::Matrix{Float64}
  end
  Base.size(o::BlockMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::BlockMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j)); return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::BlockMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs); buf[k] = o.data[i, j]; end; return buf
  end

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  opts_plain = ALPACAOptions(tol=1e-8, symmetry=:symmetric)
  opts_qrd   = ALPACAOptions(tol=1e-8, symmetry=:symmetric, qr=true)

  mat = BlockMatrix(A)
  r_plain = alpaca(mat; options=opts_plain, principal=bad_principal)
  r_qrd   = qrdalpaca(mat; options=opts_qrd, principal=bad_principal)

  @test length(r_qrd.pivot_indices) > length(r_plain.pivot_indices)
  @test length(r_qrd.pivot_indices) == n
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: hermitian block diagonal with bad principal" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  # Complex hermitian block diagonal with bad principal
  # Covers the hermitian _recon_coeffs path (L_S' adjoint) and _symmetric_refactorize
  n = 20
  B1 = randn(ComplexF64, 10, 10); B1 = B1' * B1 + 10I
  B2 = randn(ComplexF64, 10, 10); B2 = 0.1 * (B2' * B2) + 0.1I
  A = zeros(ComplexF64, n, n)
  A[1:10, 1:10] .= B1
  A[11:20, 11:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  opts_plain = ALPACAOptions(tol=1e-8, symmetry=:hermitian)
  opts_qrd   = ALPACAOptions(tol=1e-8, symmetry=:hermitian, qr=true)

  mat = DenseALPACAMatrix(A)
  r_plain = alpaca(mat; options=opts_plain, principal=bad_principal)
  r_qrd   = qrdalpaca(mat; options=opts_qrd, principal=bad_principal)

  @test length(r_qrd.pivot_indices) > length(r_plain.pivot_indices)
  @test length(r_qrd.pivot_indices) == n
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: max_rank early return" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(700)
  A = randn(10, 10); A = A * A'
  # ALPACA finds ≥3 pivots which equals max_rank, so qrdalpaca returns early
  result = qrdalpaca(A; tol=1e-12, symmetry=:symmetric, max_rank=3)
  @test length(result.pivot_indices) == 3
end

# ──────────────────────────────────────────────────────────────────
# General rectangular (exercises row cache growth)
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca: general rectangular m > n" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(641)
  m, n = 12, 6
  A = randn(m, n)

  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :general
  @test length(result.row_pivots) > 0
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "alpaca: general rectangular m < n" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(642)
  m, n = 6, 12
  A = randn(m, n)

  result = alpaca(A; tol=1e-10)
  @test result.symmetry == :general
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "lpaca: general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(643)
  n = 8
  A = randn(n, n)

  result = lpaca(A; tol=1e-10, symmetry=:general)
  @test result.symmetry == :general
  @test length(result.row_pivots) > 0
  @test norm(A - reconstruct(result)) / norm(A) < 1e-3
end

# ──────────────────────────────────────────────────────────────────
# Large enough matrix to trigger cache capacity growth
# ──────────────────────────────────────────────────────────────────

@testitem "Cache: capacity growth via large symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(651)
  # INITIAL_CAPACITY is 16, so a full-rank 24×24 forces growth
  n = 24
  A = randn(n, n)
  A = A' * A + 0.1I
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-12)
  @test length(result.pivot_indices) > 16  # must have grown past initial capacity
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Cache: capacity growth via large general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(652)
  n = 24
  A = randn(n, n)

  result = alpaca(A; tol=1e-12, symmetry=:general)
  @test length(result.pivot_indices) > 16
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# access.jl: MethodError fallbacks for AbstractALPACAMatrix
# ──────────────────────────────────────────────────────────────────

@testitem "Access: column! fallback throws MethodError" begin
  using ALPACADecomposition

  struct BareMatrix <: AbstractALPACAMatrix end
  Base.size(::BareMatrix) = (3, 3)

  buf = zeros(3)
  @test_throws MethodError column!(buf, BareMatrix(), 1)
end

@testitem "Access: row! fallback throws MethodError" begin
  using ALPACADecomposition

  struct BareMatrix2 <: AbstractALPACAMatrix end
  Base.size(::BareMatrix2) = (3, 3)

  buf = zeros(3)
  @test_throws MethodError row!(buf, BareMatrix2(), 1)
end

@testitem "Access: elements! fallback throws MethodError" begin
  using ALPACADecomposition

  struct BareMatrix3 <: AbstractALPACAMatrix end
  Base.size(::BareMatrix3) = (3, 3)

  buf = zeros(1)
  @test_throws MethodError elements!(buf, BareMatrix3(), [(1, 1)])
end

# ──────────────────────────────────────────────────────────────────
# General matrix with principal triples (covers update_principal_residuals_general!)
# ──────────────────────────────────────────────────────────────────

@testitem "alpaca: general with principal triples" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(710)
  n = 15
  A = randn(n, n)
  # Off-diagonal principal triples for general matrix
  triples = [(1, 2), (3, 5), (2, 4), (4, 1), (5, 3)]
  desc = principal_triples(triples)
  result = alpaca(A; tol=1e-10, symmetry=:general, principal=desc)
  @test result.symmetry == :general
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Large block diagonal: triggers sampling path (n_sample < nn) in qrdalpaca
# ──────────────────────────────────────────────────────────────────

@testitem "qrdalpaca: large block diagonal triggers sampling path" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  # n=100 with 10-dim strong block and 90-dim weak block.
  # After ALPACA finds ~10 pivots (block 1), nn = ~90 non-pivots.
  # n_sample = min(90, max(isqrt(90),32)) = 32 < 90 → sampling path L145.
  n = 100
  B1 = randn(10, 10); B1 = B1' * B1 + 10I
  B2 = randn(90, 90); B2 = 0.1 * (B2' * B2) + 0.1I
  A = zeros(n, n)
  A[1:10, 1:10] .= B1
  A[11:100, 11:100] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  opts = ALPACAOptions(tol=1e-8, symmetry=:symmetric, qr=true)

  mat = DenseALPACAMatrix(A)
  r = qrdalpaca(mat; options=opts, principal=bad_principal)
  @test length(r.pivot_indices) == n
  @test norm(A - reconstruct(r)) / norm(A) < 1e-6
end

# ──────────────────────────────────────────────────────────────────
# Cache capacity growth: large full-rank matrix exceeds INITIAL_CAPACITY
# ──────────────────────────────────────────────────────────────────

@testitem "Cache: col capacity growth via large full-rank symmetric" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(800)
  # INITIAL_CAPACITY = 256, so a 300×300 full-rank matrix forces growth
  n = 300
  V = randn(n, n)
  A = V' * V + I
  A = 0.5 * (A + A')

  result = alpaca(A; tol=1e-12)
  @test length(result.pivot_indices) > 256
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end

@testitem "Cache: row capacity growth via large full-rank general" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(801)
  n = 300
  A = randn(n, n) + I

  result = alpaca(A; tol=1e-12, symmetry=:general)
  @test length(result.pivot_indices) > 256
  @test norm(A - reconstruct(result)) / norm(A) < 1e-6
end
