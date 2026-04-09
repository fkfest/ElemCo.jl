#!/usr/bin/env julia
#
# Killer benchmark: large rectangular matrix with block structure,
# exponentially decaying singular values, and localized components.
#
# Matrix: 100_000 × 10_000, rank 800
# Compares: ALPACA, LPACA, ACA, LLAMA (all via SVD extraction)
# Tolerances: 1e-5, 1e-6, 1e-8
#
# Usage: julia --project=lib/ALPACADecomposition benchmark/killer_rectangular.jl

using LinearAlgebra
using Random
using Printf
using ALPACADecomposition

# ═══════════════════════════════════════════════════════════════════
# Matrix construction
# ═══════════════════════════════════════════════════════════════════

"""
    build_killer_matrix(rng; m=100_000, n=10_000, r=800)

Construct a rank-`r` matrix A = U * Diagonal(σ) * V' with:
- Exponentially decaying singular values spanning ~10 orders of magnitude
- Block-localized singular vectors (some globally spread, some concentrated
  in small row/column subsets)
- Overlapping block supports to prevent easy block-diagonal detection

Returns `(U, singular_values, V)` where `A = U * Diagonal(σ) * V'`.
The full matrix is never materialized.
"""
function build_killer_matrix(rng::AbstractRNG;
                             m::Int=100_000, n::Int=10_000, r::Int=800)
  # ── Singular values: exponential decay from 1.0 to ~1e-10 ──
  true_sv = exp.(range(log(1.0), log(1e-10), length=r))

  # ── Block structure for left singular vectors (U: m × r) ──
  # Each block has vectors supported on a specific row range.
  # Blocks overlap to create cross-talk and confuse greedy pivoting.
  blocks_U = [
    # (row_range, column_range_in_U) — describes where each set of
    # singular vectors has its support
    (1:m,             1:200),     # Block 1: globally spread (easy)
    (1:60_000,        201:400),   # Block 2: top 60% of rows
    (30_001:80_000,   401:550),   # Block 3: middle rows (overlaps 1,2)
    (70_001:85_000,   551:650),   # Block 4: narrower range
    (85_001:95_000,   651:750),   # Block 5: localized
    (95_001:96_000,   751:800),   # Block 6: very localized (1000 rows)
  ]

  # ── Block structure for right singular vectors (V: n × r) ──
  blocks_V = [
    (1:n,           1:200),     # Block 1: globally spread
    (1:5_000,       201:400),   # Block 2: first half of columns
    (3_001:7_000,   401:550),   # Block 3: overlapping
    (5_001:8_000,   551:650),   # Block 4: mid-right columns
    (7_001:9_500,   651:750),   # Block 5: right columns
    (8_001:8_800,   751:800),   # Block 6: very localized (800 cols)
  ]

  # Build raw U with block support
  U_raw = zeros(m, r)
  for (rows, cols) in blocks_U
    nr = length(rows)
    nc = length(cols)
    U_raw[rows, cols] .= randn(rng, nr, nc) ./ sqrt(nr)
  end

  # Build raw V with block support
  V_raw = zeros(n, r)
  for (rows, cols) in blocks_V
    nr = length(rows)
    nc = length(cols)
    V_raw[rows, cols] .= randn(rng, nr, nc) ./ sqrt(nr)
  end

  # Orthogonalize via QR (preserves approximate localization structure)
  println("  Orthogonalizing U ($m × $r)...")
  U = Matrix(qr(U_raw).Q)[:, 1:r]
  println("  Orthogonalizing V ($n × $r)...")
  V = Matrix(qr(V_raw).Q)[:, 1:r]

  return U, true_sv, V
end

# ═══════════════════════════════════════════════════════════════════
# Matrix-free wrapper
# ═══════════════════════════════════════════════════════════════════

"""
    FactoredMatrix(U, s, V)

Matrix-free representation of A = U * Diagonal(s) * V'.
Columns and rows are computed on-the-fly without materializing the full matrix.
"""
struct FactoredMatrix{T} <: AbstractALPACAMatrix{T}
  U::Matrix{T}    # m × r
  s::Vector{T}    # r
  V::Matrix{T}    # n × r
  m::Int
  n::Int
  r::Int
end

function FactoredMatrix(U::Matrix{T}, s::AbstractVector, V::Matrix{T}) where T
  m, r = size(U)
  n, r2 = size(V)
  @assert r == r2 == length(s)
  FactoredMatrix{T}(U, T.(s), V, m, n, r)
end

Base.size(A::FactoredMatrix) = (A.m, A.n)

function ALPACADecomposition.column!(buf::AbstractVector, A::FactoredMatrix, j::Integer)
  # A[:,j] = U * Diagonal(s) * V[j,:]' = U * (s .* V[j,:])
  # = sum_k U[:,k] * s[k] * V[j,k]
  sv = A.s .* @view(A.V[j, :])  # r-vector
  mul!(buf, A.U, sv)
  return buf
end

function ALPACADecomposition.row!(buf::AbstractVector, A::FactoredMatrix, i::Integer)
  # A[i,:] = (U[i,:] .* s') * V' → V * (s .* U[i,:])
  su = A.s .* @view(A.U[i, :])  # r-vector
  mul!(buf, A.V, su)
  return buf
end

function ALPACADecomposition.elements!(buf::AbstractVector, A::FactoredMatrix,
                                       pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  @inbounds for idx in eachindex(pairs)
    i, j = pairs[idx]
    val = zero(eltype(A.U))
    for k in 1:A.r
      val += A.U[i, k] * A.s[k] * A.V[j, k]
    end
    buf[idx] = val
  end
  return buf
end

# ═══════════════════════════════════════════════════════════════════
# Principal element selection
# ═══════════════════════════════════════════════════════════════════

"""
    smart_principal_pairs(m, n, rng; n_diag=min(m,n), n_random=2000)

Generate principal element pairs that combine diagonal elements
with random off-diagonal samples from different block regions.
"""
function smart_principal_pairs(m::Int, n::Int, rng::AbstractRNG;
                               n_diag::Int=min(m, n),
                               n_random::Int=2000)
  pairs = Tuple{Int,Int}[]

  # Diagonal elements
  for k in 1:n_diag
    push!(pairs, (k, k))
  end

  # Random off-diagonal samples from different regions of the matrix
  # This helps detect hidden off-diagonal structure
  regions = [
    (1:m÷2, 1:n÷2),
    (m÷2+1:m, 1:n÷2),
    (1:m÷2, n÷2+1:n),
    (m÷2+1:m, n÷2+1:n),
    (3*m÷4:m, 3*n÷4:n),
    (9*m÷10:m, 4*n÷5:n),
  ]
  per_region = n_random ÷ length(regions)
  for (rows, cols) in regions
    for _ in 1:per_region
      i = rand(rng, rows)
      j = rand(rng, cols)
      push!(pairs, (i, j))
    end
  end

  return unique(pairs)
end

# ═══════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════

function check_svd_accuracy(U_approx, S_approx, Vt_approx,
                            A_factored::FactoredMatrix, true_sv;
                            tol::Float64, label::String)
  r_found = length(S_approx)
  r_true = count(>(tol), true_sv)

  # Check rank
  rank_ok = r_found == r_true
  rank_str = rank_ok ? "✓" : "✗"

  # Compare singular values (sort both descending for alignment)
  s_true_above = sort(true_sv[true_sv .> tol], rev=true)
  s_approx_sorted = sort(S_approx, rev=true)

  n_compare = min(length(s_true_above), length(s_approx_sorted))
  if n_compare > 0
    max_sv_err = maximum(abs.(s_true_above[1:n_compare] .- s_approx_sorted[1:n_compare]))
    rel_sv_err = maximum(abs.(s_true_above[1:n_compare] .- s_approx_sorted[1:n_compare]) ./
                         s_true_above[1:n_compare])
  else
    max_sv_err = NaN
    rel_sv_err = NaN
  end

  # Sampled element-wise reconstruction error
  rng_sample = MersenneTwister(123)
  n_samples = 10_000
  m_full, n_full = size(A_factored)
  max_elem_err = 0.0
  buf = Vector{Float64}(undef, 1)
  for _ in 1:n_samples
    i = rand(rng_sample, 1:m_full)
    j = rand(rng_sample, 1:n_full)
    # True element
    a_true = zero(Float64)
    @inbounds for k in 1:A_factored.r
      a_true += A_factored.U[i, k] * A_factored.s[k] * A_factored.V[j, k]
    end
    # Approximate element
    if r_found > 0
      a_approx = dot(@view(U_approx[i, :]), S_approx .* @view(Vt_approx[:, j]))
    else
      a_approx = 0.0
    end
    max_elem_err = max(max_elem_err, abs(a_true - a_approx))
  end

  @printf("  %-12s │ rank: %4d/%4d %s │ |Δσ|: %8.1e │ |Δσ/σ|: %8.1e │ elem: %8.1e\n",
          label, r_found, r_true, rank_str, max_sv_err, rel_sv_err, max_elem_err)
  return (rank=r_found, rank_true=r_true, max_sv_err=max_sv_err,
          rel_sv_err=rel_sv_err, elem_err=max_elem_err)
end

function run_benchmark(A_factored::FactoredMatrix, true_sv::Vector{Float64},
                       tol::Float64, principal_pairs;
                       skip_qrdalpaca::Bool=false)
  m, n = size(A_factored)
  # Compute d_row for LLAMA (and m_eff reporting)
  d_row = vec(sum(abs2, A_factored.U .* A_factored.s', dims=2))
  d_max = maximum(d_row)
  m_eff = d_max > 0 ? max(1.0, sum(d_row) / d_max) : Float64(m)

  println("\n┌──────────────────────────────────────────────────────────────────────────────────")
  @printf("│ Tolerance: %.0e   (ALPACA pivotol = tol/√m = %.2e,  LLAMA pivotol = tol/√m_eff = %.2e)\n",
          tol, tol/sqrt(m), tol/sqrt(m_eff))
  @printf("│ m_eff = %.1f  (m = %d,  ratio = %.2f%%)\n", m_eff, m, 100*m_eff/m)
  r_true = count(>(tol), true_sv)
  println("│ True rank at this tolerance: $r_true")
  println("├──────────────────────────────────────────────────────────────────────────────────")

  results = Dict{String, Any}()
  opts_general = ALPACAOptions(tol=tol, symmetry=:general)

  methods = [
    ("ALPACA", () -> alpaca_svd(A_factored; principal=principal_pairs, options=opts_general)),
    ("LPACA",  () -> lpaca_svd(A_factored; principal=principal_pairs, options=opts_general)),
    ("ACA",    () -> alpaca_svd(A_factored; principal=Tuple{Int,Int}[], options=opts_general)),
  ]
  if !skip_qrdalpaca
    push!(methods, ("QRdALPACA", () -> qrdalpaca_svd(A_factored; principal=principal_pairs, options=opts_general)))
  end

  for (label, run_fn) in methods
    print("  Running $label... ")
    t = @elapsed begin
      res = run_fn()
    end
    @printf("%.2f s\n", t)
    results[lowercase(label)] = check_svd_accuracy(
      res.U, res.S, res.Vt, A_factored, true_sv; tol, label)
    results[lowercase(label)] = merge(results[lowercase(label)], (time=t,))
  end

  # LLAMA (separate: uses d_row, different API)
  print("  Running LLAMA... ")
  t = @elapsed begin
    res = llama_svd(A_factored; d_row, tol)
  end
  @printf("%.2f s\n", t)
  results["llama"] = check_svd_accuracy(
    res.U, res.S, res.Vt, A_factored, true_sv; tol, label="LLAMA")
  results["llama"] = merge(results["llama"], (time=t,))

  println("└──────────────────────────────────────────────────────────────────────────────────")
  return results
end


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

function main()
  m, n, r = 100_000, 10_000, 800
  rng = MersenneTwister(42)

  println("=" ^ 77)
  println("KILLER BENCHMARK: Large Rectangular Matrix")
  println("  Size: $m × $n, rank: $r")
  println("  Exponential SV decay: 1.0 → 1e-10")
  println("  Block-localized singular vectors with overlapping support")
  println("=" ^ 77)

  println("\nConstructing test matrix (factored form)...")
  t_build = @elapsed begin
    U, true_sv, V = build_killer_matrix(rng; m, n, r)
  end
  @printf("Matrix construction: %.2f s\n", t_build)
  @printf("Factor storage: U = %.1f MB, V = %.1f MB\n",
          sizeof(U) / 1e6, sizeof(V) / 1e6)

  # Report singular value spectrum
  println("\nSingular value spectrum:")
  for (i, label) in [(1, "σ₁"), (100, "σ₁₀₀"), (200, "σ₂₀₀"),
                      (400, "σ₄₀₀"), (600, "σ₆₀₀"), (800, "σ₈₀₀")]
    @printf("  %s = %.6e\n", label, true_sv[i])
  end

  # Count true ranks at each tolerance
  for tol in [1e-5, 1e-6, 1e-8]
    r_true = count(>(tol), true_sv)
    @printf("  Rank at tol=%.0e: %d\n", tol, r_true)
  end

  # Create matrix-free wrapper
  A = FactoredMatrix(U, true_sv, V)

  # Generate smart principal pairs
  println("\nGenerating principal element pairs...")
  principal_pairs = smart_principal_pairs(m, n, rng)
  println("  $(length(principal_pairs)) principal pairs (diagonal + random off-diagonal)")

  # Run benchmarks at each tolerance
  all_results = Dict{Float64, Any}()
  for tol in [1e-5, 1e-6, 1e-8]
    skip_qr = (tol <= 1e-8)  # QRdALPACA too slow at tight tolerances
    all_results[tol] = run_benchmark(A, true_sv, tol, principal_pairs;
                                     skip_qrdalpaca=skip_qr)
  end

  # ── Summary table ──
  println("\n" * "=" ^ 90)
  println("SUMMARY")
  println("=" ^ 90)
  methods = ["alpaca", "lpaca", "aca", "qrdalpaca", "llama"]
  labels  = ["ALPACA", "LPACA", "ACA", "QRdALPACA", "LLAMA"]
  @printf("%-8s │ %-6s │", "tol", "metric")
  for l in labels
    @printf(" %9s │", l)
  end
  println()
  println("─" ^ 90)
  for tol in [1e-5, 1e-6, 1e-8]
    res = all_results[tol]
    r_true = count(>(tol), true_sv)
    @printf("%.0e   │ rank   │", tol)
    for m in methods
      if haskey(res, m)
        @printf(" %4d/%4d │", res[m].rank, r_true)
      else
        @printf("     ---  │")
      end
    end
    println()
    @printf("         │ |Δσ/σ| │")
    for m in methods
      if haskey(res, m)
        @printf(" %9.2e │", res[m].rel_sv_err)
      else
        @printf("      --- │")
      end
    end
    println()
    @printf("         │ elem   │")
    for m in methods
      if haskey(res, m)
        @printf(" %9.2e │", res[m].elem_err)
      else
        @printf("      --- │")
      end
    end
    println()
    @printf("         │ time   │")
    for m in methods
      if haskey(res, m)
        @printf(" %7.2f s │", res[m].time)
      else
        @printf("      --- │")
      end
    end
    println()
    tol != 1e-8 && println("─" ^ 90)
  end
  println("=" ^ 90)
end

main()
