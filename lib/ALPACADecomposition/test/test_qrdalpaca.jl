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

# ──────────────────────────────────────────────────────────────────
# QR refinement tests: scenarios that force the QR refinement loop
# ──────────────────────────────────────────────────────────────────
# Strategy: block-diagonal matrix with a zero-gap separating two blocks.
# Bad principal covers only block 1 → ALPACA finds block 1, cold-start
# hits the zero gap and stops → QR refinement discovers block 2.
# This exercises: _fetch_elements!, _symmetric_refactorize,
# _general_refactorize, the batched QR loop, norm updates, etc.

@testitem "qrdalpaca: symmetric QR refinement via zero-gap" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  n = 20
  A = zeros(n, n)
  B1 = randn(10, 10); B1 = B1' * B1 + 10I
  B2 = randn(5, 5); B2 = B2' * B2 + 0.5I
  A[1:10, 1:10] .= B1
  A[16:20, 16:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = DenseALPACAMatrix(A)

  # Plain ALPACA misses block 2 (cold-start hits zero gap)
  r_plain = alpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric),
                   principal=bad_principal)
  @test length(r_plain.pivot_indices) == 10

  # QRD finds both blocks
  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 15
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: hermitian QR refinement via zero-gap" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  n = 20
  A = zeros(ComplexF64, n, n)
  B1 = randn(ComplexF64, 10, 10); B1 = B1' * B1 + 10I
  B2 = randn(ComplexF64, 5, 5); B2 = B2' * B2 + 0.5I
  A[1:10, 1:10] .= B1
  A[16:20, 16:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:hermitian, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 15
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: general QR refinement via zero-gap" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  n = 20
  A = zeros(n, n)
  B1 = randn(10, 10)
  B2 = randn(5, 5)
  A[1:10, 1:10] .= B1
  A[16:20, 16:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = DenseALPACAMatrix(A)

  r_plain = alpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:general),
                   principal=bad_principal)
  @test length(r_plain.pivot_indices) == 10

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:general, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 15
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: matrix-free symmetric QR refinement" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  struct QRDSymBlockMatrix <: AbstractALPACAMatrix{Float64}
    data::Matrix{Float64}
  end
  Base.size(o::QRDSymBlockMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::QRDSymBlockMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j)); return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::QRDSymBlockMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs); buf[k] = o.data[i, j]; end; return buf
  end

  n = 20
  A = zeros(n, n)
  B1 = randn(10, 10); B1 = B1' * B1 + 10I
  B2 = randn(5, 5); B2 = B2' * B2 + 0.5I
  A[1:10, 1:10] .= B1
  A[16:20, 16:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = QRDSymBlockMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 15
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: matrix-free general QR refinement" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(42)

  struct QRDGenBlockMatrix <: AbstractALPACAMatrix{Float64}
    data::Matrix{Float64}
  end
  Base.size(o::QRDGenBlockMatrix) = size(o.data)
  function ALPACADecomposition.column!(buf::AbstractVector, o::QRDGenBlockMatrix, j::Integer)
    copyto!(buf, view(o.data, :, j)); return buf
  end
  function ALPACADecomposition.row!(buf::AbstractVector, o::QRDGenBlockMatrix, i::Integer)
    copyto!(buf, view(o.data, i, :)); return buf
  end
  function ALPACADecomposition.elements!(buf::AbstractVector, o::QRDGenBlockMatrix,
      pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
    for (k, (i, j)) in enumerate(pairs); buf[k] = o.data[i, j]; end; return buf
  end

  n = 20
  A = zeros(n, n)
  B1 = randn(10, 10)
  B2 = randn(5, 5)
  A[1:10, 1:10] .= B1
  A[16:20, 16:20] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = QRDGenBlockMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:general, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 15
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: many hidden columns (batched QR)" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(43)

  # Larger matrix with many hidden columns to exercise batch processing
  n = 60
  A = zeros(n, n)
  B1 = randn(20, 20); B1 = B1' * B1 + 10I
  B2 = randn(15, 15); B2 = B2' * B2 + I
  B3 = randn(10, 10); B3 = B3' * B3 + 0.1I
  A[1:20, 1:20] .= B1
  A[26:40, 26:40] .= B2
  A[51:60, 51:60] .= B3

  bad_principal = principal_pairs([(i, i) for i in 1:20])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 45
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: complex symmetric QR refinement" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(44)

  n = 16
  A = zeros(ComplexF64, n, n)
  B1 = randn(ComplexF64, 8, 8); B1 = B1 * transpose(B1) + I
  B1 = 0.5 * (B1 + transpose(B1))
  B2 = randn(ComplexF64, 4, 4); B2 = B2 * transpose(B2) + 0.5I
  B2 = 0.5 * (B2 + transpose(B2))
  A[1:8, 1:8] .= B1
  A[13:16, 13:16] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:8])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 12
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: many blocks (multi-round QR refinement)" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(45)

  # 5 blocks separated by zero gaps → multiple QR refinement rounds
  # This exercises the batched QR loop, norm updates, and capacity expansion
  n = 100
  A = zeros(n, n)
  block_ranges = [1:10, 21:30, 41:55, 61:70, 81:95]
  total_rank = sum(length, block_ranges)
  for (i, rng) in enumerate(block_ranges)
    bsz = length(rng)
    B = randn(bsz, bsz)
    B = B' * B + (11 - 2i) * I  # decreasing conditioning
    A[rng, rng] .= B
  end

  # Principal covers only the first block
  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == total_rank
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: subsampling branch (large non-pivot set)" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(46)

  # Need > 700 non-pivot columns after ALPACA to trigger subsampling
  # Block-diagonal: block 1 (indices 1:10) covered by principal,
  # zero gap (indices 11:750), block 2 (indices 751:760)
  n = 760
  A = zeros(n, n)
  B1 = randn(10, 10); B1 = B1' * B1 + 10I
  B2 = randn(10, 10); B2 = B2' * B2 + I
  A[1:10, 1:10] .= B1
  A[751:760, 751:760] .= B2

  bad_principal = principal_pairs([(i, i) for i in 1:10])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:symmetric, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == 20
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end

@testitem "qrdalpaca: general with many blocks" setup=[Helpers] begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(47)

  # Multiple blocks with general (non-symmetric) entries
  # Tests _general_refactorize with multiple refinement pivots
  n = 60
  A = zeros(n, n)
  block_ranges = [1:12, 21:32, 41:48, 51:60]
  total_rank = sum(length, block_ranges)
  for rng in block_ranges
    bsz = length(rng)
    A[rng, rng] .= randn(bsz, bsz)
  end

  bad_principal = principal_pairs([(i, i) for i in 1:12])
  mat = DenseALPACAMatrix(A)

  r_qrd = qrdalpaca(mat; options=ALPACAOptions(tol=1e-10, symmetry=:general, qr=true),
                    principal=bad_principal)
  @test length(r_qrd.pivot_indices) == total_rank
  @test norm(A - reconstruct(r_qrd)) / norm(A) < 1e-6
end
