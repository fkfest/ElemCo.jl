@testitem "llama: real low-rank matrix" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(501)

  n, m, r = 20, 40, 4
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([10.0, 5.0, 2.0, 1.0])
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * S0 * V0'

  d_row = vec(sum(abs2, A, dims=2))

  result = llama(A; tol=1e-10, d_row)
  Q = result.Q

  # Projector comparison: ‖QQ' - U₀U₀'‖_F should be small
  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: real low-rank via dense convenience" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(502)

  n, m, r = 15, 30, 3
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([8.0, 3.0, 1.0])
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * S0 * V0'

  # Dense convenience: d_row auto-computed
  result = llama(A; tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: singular value recovery" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(503)

  n, m, r = 20, 40, 3
  true_sv = [10.0, 5.0, 1.0]
  U0 = Matrix(qr(randn(n, r)).Q)
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  result = llama(A; tol=1e-10)
  # Singular values should be approximately recovered
  @test length(result.singular_values) == r
  sv_sorted = sort(result.singular_values, rev=true)
  for (s_approx, s_true) in zip(sv_sorted, true_sv)
    @test abs(s_approx - s_true) / s_true < 1e-4
  end
end

@testitem "llama: complex low-rank matrix" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(504)

  n, m, r = 16, 32, 3
  U0 = Matrix(qr(randn(ComplexF64, n, r)).Q)
  S0 = Diagonal([7.0, 3.0, 1.5])
  V0 = Matrix(qr(randn(ComplexF64, m, r)).Q)
  A = U0 * S0 * V0'

  result = llama(A; tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: rank-1 matrix" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(505)

  n, m = 10, 20
  u = normalize(randn(n))
  v = normalize(randn(m))
  A = 5.0 * u * v'

  result = llama(A; tol=1e-10)
  Q = result.Q
  @test size(Q, 2) == 1
  # Q should span the same direction as u
  @test abs(abs(dot(Q[:, 1], u)) - 1.0) < 1e-8
end

@testitem "llama: zero matrix" begin
  using ALPACADecomposition
  using LinearAlgebra

  A = zeros(8, 12)
  result = llama(A; tol=1e-10)
  @test size(result.Q, 2) == 0
  @test isempty(result.singular_values)
end

@testitem "llama: max_rank constraint" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(506)

  n, m, r = 20, 40, 5
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal(Float64.(r:-1:1))
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * S0 * V0'

  result = llama(A; tol=1e-10, max_rank=3)
  @test size(result.Q, 2) <= 3
end

@testitem "llama: matrix-free interface" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(507)

  n, m, r = 12, 24, 3
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([6.0, 3.0, 1.0])
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * S0 * V0'

  mat = DenseALPACAMatrix(A)
  d_row = vec(sum(abs2, A, dims=2))

  result = llama(mat; d_row, tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
end

@testitem "llama: access counting" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(508)

  # Custom matrix wrapper that counts accesses
  mutable struct CountingMatrix{T} <: AbstractALPACAMatrix{T}
    data::Matrix{T}
    col_accesses::Int
    row_accesses::Int
  end
  CountingMatrix(A::Matrix{T}) where T = CountingMatrix{T}(A, 0, 0)
  Base.size(m::CountingMatrix) = size(m.data)
  function ALPACADecomposition.column!(buf::AbstractVector, m::CountingMatrix, j::Integer)
    m.col_accesses += 1
    copyto!(buf, view(m.data, :, j))
    return buf
  end
  function ALPACADecomposition.row!(buf::AbstractVector, m::CountingMatrix, i::Integer)
    m.row_accesses += 1
    copyto!(buf, view(m.data, i, :))
    return buf
  end

  n, m_dim, r = 20, 40, 4
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([10.0, 5.0, 2.0, 1.0])
  V0 = Matrix(qr(randn(m_dim, r)).Q)
  A = U0 * S0 * V0'

  mat = CountingMatrix(A)
  d_row = vec(sum(abs2, A, dims=2))

  result = llama(mat; d_row, tol=1e-10)
  Q = result.Q

  # Should find the correct rank
  @test size(Q, 2) == r
  # Column accesses = exactly r (one per successful pivot)
  @test mat.col_accesses <= 3 * r  # generous upper bound
  # Row accesses can be up to m (exhausted rows are tried then skipped)
  @test mat.row_accesses <= n
end

@testitem "llama_svd: basic" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(509)

  n, m, r = 15, 30, 3
  true_sv = [8.0, 4.0, 1.0]
  U0 = Matrix(qr(randn(n, r)).Q)
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  U, S, Vt = llama_svd(A; tol=1e-10)
  @test length(S) == r
  # U should span the column space
  P_approx = U * U'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
end

@testitem "llama: similar column norms, diverse directions" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(510)

  # Pathological: all columns have similar norms but span 3 directions
  n, m, r = 12, 36, 3
  U0 = Matrix(qr(randn(n, r)).Q)
  # All singular values equal → all column norms similar
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * (5.0 * I(r)) * V0'

  result = llama(A; tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: fullsvd returns right singular vectors" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(520)

  n, m, r = 20, 40, 4
  true_sv = [10.0, 5.0, 2.0, 1.0]
  U0 = Matrix(qr(randn(n, r)).Q)
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  result = llama(A; tol=1e-10, fullsvd=true)
  @test result.V !== nothing
  @test size(result.V) == (m, r)

  # Full approximation: A ≈ Q * diag(σ) * V'
  A_approx = result.Q * Diagonal(result.singular_values) * result.V'
  @test norm(A - A_approx) / norm(A) < 1e-6

  # V should be approximately orthonormal
  @test norm(result.V' * result.V - I(r)) < 1e-6

  # Right-space projector should match
  P_V = result.V * result.V'
  P_exact = V0 * V0'
  @test norm(P_V - P_exact) < 1e-6
end

@testitem "llama: fullsvd=false returns V=nothing" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(521)

  A = randn(15, 30) * randn(30, 30)
  result = llama(A; tol=1e-6)
  @test result.V === nothing
end

@testitem "llama: fullsvd with complex matrix" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(522)

  n, m, r = 20, 40, 3
  true_sv = [8.0, 3.0, 1.0]
  U0 = Matrix(qr(randn(ComplexF64, n, r)).Q)
  V0 = Matrix(qr(randn(ComplexF64, m, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  result = llama(A; tol=1e-10, fullsvd=true)
  @test result.V !== nothing
  @test size(result.V) == (m, r)

  A_approx = result.Q * Diagonal(result.singular_values) * result.V'
  @test norm(A - A_approx) / norm(A) < 1e-6
end

@testitem "llama_svd returns actual Vt" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(523)

  n, m, r = 20, 40, 4
  true_sv = [10.0, 5.0, 2.0, 1.0]
  U0 = Matrix(qr(randn(n, r)).Q)
  V0 = Matrix(qr(randn(m, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  U, S, Vt = llama_svd(A; tol=1e-10)
  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-6
  @test size(Vt) == (r, m)
end

# ──────────────────────────────────────────────────────────────────
# Column-guided (d_col) tests
# ──────────────────────────────────────────────────────────────────

@testitem "llama: d_col gives same column-space basis as d_row" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(530)

  m, n, r = 20, 40, 4
  U0 = Matrix(qr(randn(m, r)).Q)
  V0 = Matrix(qr(randn(n, r)).Q)
  A = U0 * Diagonal([10.0, 5.0, 2.0, 1.0]) * V0'

  d_row = vec(sum(abs2, A, dims=2))
  d_col = vec(sum(abs2, A, dims=1))

  r_row = llama(A; d_row, tol=1e-10)
  r_col = llama(A; d_col, tol=1e-10)

  # Both should give equally good column-space bases
  proj_err_row = maximum(abs, A - r_row.Q * (r_row.Q' * A))
  proj_err_col = maximum(abs, A - r_col.Q * (r_col.Q' * A))
  @test proj_err_row < 1e-10
  @test proj_err_col < 1e-10
  @test size(r_col.Q) == size(r_row.Q)
  @test length(r_col.singular_values) == length(r_row.singular_values)
end

@testitem "llama: d_col with fullsvd" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(531)

  m, n, r = 20, 40, 4
  true_sv = [10.0, 5.0, 2.0, 1.0]
  U0 = Matrix(qr(randn(m, r)).Q)
  V0 = Matrix(qr(randn(n, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  d_col = vec(sum(abs2, A, dims=1))
  result = llama(A; d_col, tol=1e-10, fullsvd=true)

  @test result.V !== nothing
  @test size(result.Q) == (m, r)
  @test size(result.V) == (n, r)

  # Full SVD reconstruction
  A_approx = result.Q * Diagonal(result.singular_values) * result.V'
  @test norm(A - A_approx) / norm(A) < 1e-10
end

@testitem "llama: d_col with complex matrix" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(532)

  m, n, r = 20, 40, 3
  true_sv = [8.0, 3.0, 1.0]
  U0 = Matrix(qr(randn(ComplexF64, m, r)).Q)
  V0 = Matrix(qr(randn(ComplexF64, n, r)).Q)
  A = U0 * Diagonal(true_sv) * V0'

  d_col = vec(sum(abs2, A, dims=1))
  result = llama(A; d_col, tol=1e-10, fullsvd=true)

  @test size(result.Q) == (m, r)
  @test size(result.V) == (n, r)

  # Verify singular values
  sv = sort(result.singular_values, rev=true)
  @test sv ≈ true_sv atol=1e-10

  # Full SVD reconstruction
  A_approx = result.Q * Diagonal(result.singular_values) * result.V'
  @test norm(A - A_approx) / norm(A) < 1e-10
end

@testitem "llama: d_col error when both d_row and d_col given" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(533)

  A = randn(10, 20)
  d_row = vec(sum(abs2, A, dims=2))
  d_col = vec(sum(abs2, A, dims=1))

  @test_throws ArgumentError llama(A; d_row, d_col, tol=1e-10)
end

# ──────────────────────────────────────────────────────────────────
# Symmetric / Hermitian wrappers
# ──────────────────────────────────────────────────────────────────

@testitem "llama: real Symmetric dense convenience" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(540)

  n, r = 30, 4
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([10.0, 5.0, 2.0, 1.0])
  A = U0 * S0 * U0'  # symmetric, rank-r
  A = Symmetric(A)

  result = llama(A; tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: real Symmetric SVD" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(541)

  n, r = 25, 3
  U0 = Matrix(qr(randn(n, r)).Q)
  true_sv = [8.0, 3.0, 1.0]
  S0 = Diagonal(true_sv)
  A = U0 * S0 * U0'
  A = Symmetric(A)

  U, S, Vt = llama_svd(A; tol=1e-10)

  sv = sort(S, rev=true)
  @test sv ≈ true_sv atol=1e-8
  @test size(U, 2) == r

  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-8
end

@testitem "llama: complex Hermitian dense convenience" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(542)

  n, r = 25, 4
  U0 = Matrix(qr(randn(ComplexF64, n, r)).Q)
  S0 = Diagonal([10.0, 5.0, 2.0, 1.0])
  A = U0 * S0 * U0'  # hermitian, rank-r
  A = Hermitian(A)

  result = llama(A; tol=1e-10)
  Q = result.Q

  P_approx = Q * Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
  @test size(Q, 2) == r
end

@testitem "llama: complex Hermitian SVD" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(543)

  n, r = 20, 3
  U0 = Matrix(qr(randn(ComplexF64, n, r)).Q)
  true_sv = [8.0, 3.0, 1.0]
  S0 = Diagonal(true_sv)
  A = U0 * S0 * U0'
  A = Hermitian(A)

  U, S, Vt = llama_svd(A; tol=1e-10)

  sv = sort(S, rev=true)
  @test sv ≈ true_sv atol=1e-8
  @test size(U, 2) == r

  A_approx = U * Diagonal(S) * Vt
  @test norm(A - A_approx) / norm(A) < 1e-8
end

@testitem "llama: SymmetricALPACAMatrix row! equals column!" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(544)

  n = 15
  A = randn(n, n)
  A = A + A'  # make symmetric
  wrapped = SymmetricALPACAMatrix(DenseALPACAMatrix(A))

  buf_col = zeros(n)
  buf_row = zeros(n)
  for j in 1:n
    column!(buf_col, wrapped, j)
    row!(buf_row, wrapped, j)
    @test buf_col == buf_row
  end

  @test issymmetric(wrapped)
  @test ishermitian(wrapped)
end

@testitem "llama: HermitianALPACAMatrix row! equals conj(column!)" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(545)

  n = 15
  A = randn(ComplexF64, n, n)
  A = A + A'  # make hermitian
  wrapped = HermitianALPACAMatrix(DenseALPACAMatrix(A))

  buf_col = zeros(ComplexF64, n)
  buf_row = zeros(ComplexF64, n)
  for j in 1:n
    column!(buf_col, wrapped, j)
    row!(buf_row, wrapped, j)
    @test buf_row == conj(buf_col)
  end

  @test !issymmetric(wrapped)
  @test ishermitian(wrapped)
end

@testitem "llama: real Hermitian treated as symmetric" begin
  using ALPACADecomposition
  using LinearAlgebra, Random
  Random.seed!(546)

  n, r = 20, 3
  U0 = Matrix(qr(randn(n, r)).Q)
  S0 = Diagonal([6.0, 3.0, 1.0])
  A = U0 * S0 * U0'
  A = Hermitian(A)

  result = llama(A; tol=1e-10)
  @test size(result.Q, 2) == r

  P_approx = result.Q * result.Q'
  P_exact = U0 * U0'
  @test norm(P_approx - P_exact) < 1e-6
end

@testitem "llama: _llama_finalize robust to rank-deficient ACA factors" begin
  # Regression: an unpivoted Cholesky of the row/column Gram matrix throws
  # PosDefException when an ACA factor is numerically rank-deficient (seen in
  # EOM-SVD-DCSD). QR-based finalization must instead recover the exact SVD.
  using ALPACADecomposition
  using LinearAlgebra

  # Row factor with a duplicated column => R^H R is singular.
  Cv = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0;
        1.0 1.0 0.0; 0.5 0.0 1.0; 0.0 2.0 1.0]   # 6×3, full column rank
  Rv = [1.0 0.0 1.0; 0.0 1.0 0.0; 2.0 0.0 2.0;
        0.0 3.0 0.0; 1.0 1.0 1.0]                # 5×3, col 3 == col 1 (rank 2)
  d  = [2.0, 1.5, 0.7]
  A  = Cv * Diagonal(d) * transpose(Rv)          # true rank 2

  @test rank(Rv) == 2                             # old unpivoted Cholesky would throw here

  for fullsvd in (false, true)
    Q, sv, V, nk = ALPACADecomposition._llama_finalize(Cv, Rv, d, 3, 1e-10, fullsvd)
    Aref = svd(A)
    @test nk == 2
    @test isapprox(sv, Aref.S[1:nk]; rtol=1e-10)
    # Column space recovered exactly
    @test norm(Q * Q' - Aref.U[:, 1:nk] * Aref.U[:, 1:nk]') < 1e-10
    if fullsvd
      @test isapprox(Q * Diagonal(sv) * V', A; atol=1e-10)
    end
  end
end

@testitem "llama: _llama_finalize robust to rank-deficient complex factors" begin
  # Complex counterpart of the rank-deficiency regression above. This is the
  # case that actually crashed in EOM-SVD-DCSD (complex integrals): for complex
  # T the finalization always takes the full path, where transpose(Rv) (not the
  # adjoint) must be used, and an unpivoted Cholesky of the singular Gram throws.
  # QR finalization must recover the exact SVD here too.
  using ALPACADecomposition
  using LinearAlgebra

  Cv = ComplexF64[1 0 0; 0 1 0; 0 0 1;
                  1+1im 1 0; 0.5 0 1; 0 2 1im]      # 6×3, full column rank
  c1 = ComplexF64[1, 0, 2-1im, 0, 1+1im]
  c2 = ComplexF64[0, 1, 0, 3, 1]
  Rv = hcat(c1, c2, 2 .* c1)                         # 5×3, col 3 == 2·col 1 (rank 2)
  d  = ComplexF64[2+1im, 1.5-0.3im, 0.7]
  A  = Cv * Diagonal(d) * transpose(Rv)              # stored convention A ≈ C D Rᵀ

  @test rank(Rv) == 2                                # old unpivoted Cholesky would throw here

  for fullsvd in (false, true)
    Q, sv, V, nk = ALPACADecomposition._llama_finalize(Cv, Rv, d, 3, 1e-10, fullsvd)
    Aref = svd(A)
    @test nk == 2
    @test isapprox(sv, Aref.S[1:nk]; rtol=1e-10)
    # Column space recovered exactly
    @test norm(Q * Q' - Aref.U[:, 1:nk] * Aref.U[:, 1:nk]') < 1e-10
    if fullsvd
      @test isapprox(Q * Diagonal(sv) * V', A; atol=1e-10)
    end
  end
end

@testitem "llama: _llama_finalize robust to rank-deficient COLUMN factor" begin
  # The column factor C can also be (numerically) rank-deficient, which makes the
  # QR R_C singular. The basis must be formed from the QR Q-factor (Q = Q_C·U_B),
  # NOT via a triangular solve `R_C \\ U`: with a singular R_C that solve either
  # throws SingularException (exactly-zero pivot) or silently returns a wrong,
  # non-orthonormal basis (here ‖QᵀQ−I‖ ≈ 0.56 with the old formula).
  using ALPACADecomposition
  using LinearAlgebra

  for T in (Float64, ComplexF64)
    c1 = T <: Real ? T[1, 0, 2, 0, 1, 0.5] : T[1, 0, 2-1im, 0, 1+1im, 0.5]
    c2 = T <: Real ? T[0, 1, 0, 3, 1, 0.0] : T[0, 1, 0, 3, 1, 1im]
    Cv = hcat(c1, c2, 2 .* c1)                       # 6×3, col 3 == 2·col 1 (rank 2)
    Rv = T <: Real ? T[1 0 0; 0 1 0; 0 0 1; 1 1 0; 0.5 0 1] :
                     T[1 0 0; 0 1 0; 0 0 1; 1+1im 1 0; 0.5 0 1im]  # 5×3, full col rank
    d  = T <: Real ? T[2.0, 1.5, 0.7] : T[2+1im, 1.5-0.3im, 0.7]
    A  = Cv * Diagonal(d) * transpose(Rv)            # true rank 2

    @test rank(Cv) == 2                              # singular R_C: `R_C \\ U` would throw

    for fullsvd in (false, true)
      Q, sv, V, nk = ALPACADecomposition._llama_finalize(Cv, Rv, d, 3, 1e-10, fullsvd)
      Aref = svd(A)
      @test nk == 2
      @test isapprox(sv, Aref.S[1:nk]; rtol=1e-10)
      @test norm(Q' * Q - I) < 1e-10                 # Q orthonormal despite singular R_C
      @test norm(Q * Q' - Aref.U[:, 1:nk] * Aref.U[:, 1:nk]') < 1e-10
      if fullsvd
        @test isapprox(Q * Diagonal(sv) * V', A; atol=1e-10)
      end
    end
  end
end
