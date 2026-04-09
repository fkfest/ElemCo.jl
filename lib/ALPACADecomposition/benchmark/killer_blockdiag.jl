#!/usr/bin/env julia
#
# Killer benchmark: block-diagonal matrix with ZERO diagonal elements.
#
# This benchmark is specifically designed to expose the limitations of
# diagonal-based pivot selection (ALPACA/ACA) on matrices where the
# diagonal carries no signal. LLAMA, which uses d_row = diag(AA') for
# row-pivot guidance, should discover all blocks while ALPACA/ACA get
# trapped in the first block found.
#
# Matrix: 100_000 × 10_000, rank 800
#   - 4 disjoint blocks (rows and columns don't overlap)
#   - All U supports start above row 10_000 → A[i,i] = 0 for all i
#   - Singular values interleaved round-robin across blocks with
#     exponential decay from 1.0 → 1e-12
#   - At any tolerance, ~25% of the needed rank is in each block
#
# Expected behavior:
#   - ALPACA: diagonal principal = 0 → immediate ACA fallback → discovers
#     only block 1 via cold start → ~25% of true rank
#   - ACA: same greedy row/column cycling stays within block 1
#   - LLAMA: d_row guides to all 4 blocks → full rank recovery
#
# Usage: julia --project=lib/ALPACADecomposition lib/ALPACADecomposition/benchmark/killer_blockdiag.jl

using LinearAlgebra
using Random
using Printf
using ALPACADecomposition

# ═══════════════════════════════════════════════════════════════════
# Matrix construction
# ═══════════════════════════════════════════════════════════════════

"""
    build_killer_blockdiag(rng; m=100_000, n=10_000, r=800, n_blocks=4)

Construct a rank-`r` block-diagonal matrix A = U * Diagonal(σ) * V' with:
- `n_blocks` disjoint blocks (non-overlapping row and column supports)
- Zero diagonal: A[i,i] = 0 for all i = 1..min(m,n)
- Singular values distributed round-robin across blocks with exponential decay
- Each block gets r÷n_blocks singular values

Returns `(U, singular_values, V, block_info)`.
"""
function build_killer_blockdiag(rng::AbstractRNG;
                                m::Int=100_000, n::Int=10_000,
                                r::Int=800, n_blocks::Int=4)
  @assert r % n_blocks == 0 "rank must be divisible by n_blocks"
  r_per_block = r ÷ n_blocks
  col_block_size = n ÷ n_blocks

  # Row block sizes: distribute rows 10001..m across blocks equally
  # (rows 1..n have U=0 to guarantee zero diagonal)
  m_avail = m - n  # rows available for U support above the diagonal region
  row_block_size = m_avail ÷ n_blocks

  # ── Singular values: round-robin across blocks ──
  # σ_k = 10^(-12 * (k-1)/(r-1)) for k = 1..r
  # Block assignment: block((k-1) mod n_blocks + 1)
  all_sv = [10.0^(-12.0 * (k - 1) / (r - 1)) for k in 1:r]

  # Assign SVs to blocks
  block_sv_indices = [Int[] for _ in 1:n_blocks]
  for k in 1:r
    b = (k - 1) % n_blocks + 1
    push!(block_sv_indices[b], k)
  end

  # ── Build block structure ──
  block_info = []
  U_raw = zeros(m, r)
  V_raw = zeros(n, r)

  for b in 1:n_blocks
    # Column range for block b
    c_start = (b - 1) * col_block_size + 1
    c_end = b * col_block_size
    col_range = c_start:c_end

    # Row range: starts at n+1 to guarantee zero diagonal
    r_start = n + (b - 1) * row_block_size + 1
    r_end = n + b * row_block_size
    row_range = r_start:r_end

    sv_indices = block_sv_indices[b]

    push!(block_info, (block=b, rows=row_range, cols=col_range,
                       n_sv=length(sv_indices),
                       sv_range=(all_sv[sv_indices[1]], all_sv[sv_indices[end]])))

    # Fill U and V with random entries in block support
    nr = length(row_range)
    nc = length(col_range)
    for (local_k, global_k) in enumerate(sv_indices)
      U_raw[row_range, global_k] .= randn(rng, nr) ./ sqrt(nr)
      V_raw[col_range, global_k] .= randn(rng, nc) ./ sqrt(nc)
    end
  end

  # Orthogonalize within each block (block-diagonal QR)
  println("  Orthogonalizing U and V (block-diagonal structure)...")
  U = zeros(m, r)
  V = zeros(n, r)
  for b in 1:n_blocks
    sv_idx = block_sv_indices[b]
    bi = block_info[b]

    # Orthogonalize U block
    U_block = U_raw[bi.rows, sv_idx]
    Q_U = Matrix(qr(U_block).Q)[:, 1:length(sv_idx)]
    U[bi.rows, sv_idx] .= Q_U

    # Orthogonalize V block
    V_block = V_raw[bi.cols, sv_idx]
    Q_V = Matrix(qr(V_block).Q)[:, 1:length(sv_idx)]
    V[bi.cols, sv_idx] .= Q_V
  end

  return U, all_sv, V, block_info
end

# ═══════════════════════════════════════════════════════════════════
# Matrix-free wrapper (same as killer_rectangular.jl)
# ═══════════════════════════════════════════════════════════════════

struct FactoredMatrix{T} <: AbstractALPACAMatrix{T}
  U::Matrix{T}
  s::Vector{T}
  V::Matrix{T}
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
  sv = A.s .* @view(A.V[j, :])
  mul!(buf, A.U, sv)
  return buf
end

function ALPACADecomposition.row!(buf::AbstractVector, A::FactoredMatrix, i::Integer)
  su = A.s .* @view(A.U[i, :])
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
# Accuracy checking
# ═══════════════════════════════════════════════════════════════════

function check_svd_accuracy(U_approx, S_approx, Vt_approx,
                            A_factored::FactoredMatrix, true_sv;
                            tol::Float64, label::String)
  r_found = length(S_approx)
  r_true = count(>(tol), true_sv)

  rank_str = r_found == r_true ? "✓" : "✗"

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
  for _ in 1:n_samples
    i = rand(rng_sample, 1:m_full)
    j = rand(rng_sample, 1:n_full)
    a_true = zero(Float64)
    @inbounds for k in 1:A_factored.r
      a_true += A_factored.U[i, k] * A_factored.s[k] * A_factored.V[j, k]
    end
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

# ═══════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════

function run_benchmark(A_factored::FactoredMatrix, true_sv::Vector{Float64},
                       tol::Float64, principal_pairs)
  m, n = size(A_factored)
  # Compute d_row for LLAMA
  d_row = vec(sum(abs2, A_factored.U .* A_factored.s', dims=2))

  r_true = count(>(tol), true_sv)

  println("\n┌──────────────────────────────────────────────────────────────────────────────────")
  @printf("│ Tolerance: %.0e   │ True rank: %d\n", tol, r_true)
  println("├──────────────────────────────────────────────────────────────────────────────────")

  results = Dict{String, Any}()
  opts_general = ALPACAOptions(tol=tol, symmetry=:general)

  methods = [
    ("ALPACA", () -> alpaca_svd(A_factored; principal=principal_pairs, options=opts_general)),
    ("LPACA",  () -> lpaca_svd(A_factored; principal=principal_pairs, options=opts_general)),
    ("ACA",    () -> alpaca_svd(A_factored; principal=Tuple{Int,Int}[], options=opts_general)),
  ]

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

  # LLAMA
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
  m, n, r, n_blocks = 100_000, 10_000, 800, 4
  rng = MersenneTwister(42)

  println("=" ^ 77)
  println("KILLER BENCHMARK: Block-Diagonal with Zero Diagonal")
  println("  Size: $m × $n, rank: $r, blocks: $n_blocks")
  println("  Zero diagonal: A[i,i] = 0 for all i")
  println("  SVs interleaved round-robin across blocks: 1.0 → 1e-12")
  println("  ALPACA/ACA expected to find only ~1 block (~25% rank)")
  println("  LLAMA expected to find all blocks (full rank)")
  println("=" ^ 77)

  println("\nConstructing test matrix (factored form)...")
  t_build = @elapsed begin
    U, true_sv, V, block_info = build_killer_blockdiag(rng; m, n, r, n_blocks)
  end
  @printf("Matrix construction: %.2f s\n", t_build)
  @printf("Factor storage: U = %.1f MB, V = %.1f MB\n",
          sizeof(U) / 1e6, sizeof(V) / 1e6)

  # Report block structure
  println("\nBlock structure:")
  for bi in block_info
    @printf("  Block %d: U rows %6d-%6d, V cols %5d-%5d, %3d SVs (σ: %.1e → %.1e)\n",
            bi.block, first(bi.rows), last(bi.rows),
            first(bi.cols), last(bi.cols),
            bi.n_sv, bi.sv_range...)
  end

  # Verify zero diagonal
  A_fm = FactoredMatrix(U, true_sv, V)
  diag_max = 0.0
  buf = Vector{Float64}(undef, 1)
  for i in 1:min(m, n)
    elements!(buf, A_fm, [(i, i)])
    diag_max = max(diag_max, abs(buf[1]))
  end
  @printf("\nDiagonal verification: max|A[i,i]| = %.2e  (should be ≈ 0)\n", diag_max)

  # Report singular value spectrum
  println("\nSingular value spectrum:")
  for (i, label) in [(1, "σ₁"), (100, "σ₁₀₀"), (200, "σ₂₀₀"),
                      (400, "σ₄₀₀"), (600, "σ₆₀₀"), (800, "σ₈₀₀")]
    b = (i - 1) % n_blocks + 1
    @printf("  %s = %.6e  (block %d)\n", label, true_sv[i], b)
  end

  # Count true ranks at each tolerance
  for tol in [1e-5, 1e-8, 1e-10]
    r_true = count(>(tol), true_sv)
    # Count per-block
    block_ranks = Int[]
    for b in 1:n_blocks
      block_r = count(k -> true_sv[k] > tol && (k - 1) % n_blocks + 1 == b, 1:r)
      push!(block_ranks, block_r)
    end
    @printf("  Rank at tol=%.0e: %d  (per block: %s)\n", tol, r_true,
            join(string.(block_ranks), "/"))
  end

  # Create matrix-free wrapper
  A = A_fm

  # Principal pairs: diagonal only (demonstrates ALPACA's weakness)
  println("\nPrincipal elements: diagonal ($(min(m,n)) pairs, all zero)")
  diagonal_pairs = [(i, i) for i in 1:min(m, n)]

  # Run benchmarks at each tolerance
  all_results = Dict{Float64, Any}()
  for tol in [1e-5, 1e-8, 1e-10]
    all_results[tol] = run_benchmark(A, true_sv, tol, diagonal_pairs)
  end

  # ── Summary table ──
  println("\n" * "=" ^ 80)
  println("SUMMARY")
  println("=" ^ 80)
  methods_list = ["alpaca", "lpaca", "aca", "llama"]
  labels_list  = ["ALPACA", "LPACA", "ACA", "LLAMA"]
  @printf("%-8s │ %-6s │", "tol", "metric")
  for l in labels_list
    @printf(" %9s │", l)
  end
  println()
  println("─" ^ 80)
  for tol in [1e-5, 1e-8, 1e-10]
    res = all_results[tol]
    r_true = count(>(tol), true_sv)
    @printf("%.0e   │ rank   │", tol)
    for m in methods_list
      @printf(" %4d/%4d │", res[m].rank, r_true)
    end
    println()
    @printf("         │ |Δσ/σ| │")
    for m in methods_list
      @printf(" %9.2e │", res[m].rel_sv_err)
    end
    println()
    @printf("         │ elem   │")
    for m in methods_list
      @printf(" %9.2e │", res[m].elem_err)
    end
    println()
    @printf("         │ time   │")
    for m in methods_list
      @printf(" %7.2f s │", res[m].time)
    end
    println()
    tol != 1e-10 && println("─" ^ 80)
  end
  println("=" ^ 80)

  # ── LLAMA advantage summary ──
  println("\nLLAMA advantage over best ACA/ALPACA variant:")
  for tol in [1e-5, 1e-8, 1e-10]
    res = all_results[tol]
    r_true = count(>(tol), true_sv)
    llama_rank = res["llama"].rank
    best_other = maximum(res[m].rank for m in ["alpaca", "lpaca", "aca"])
    @printf("  tol=%.0e: LLAMA %4d vs best-other %4d  (%.0f%% vs %.0f%% of true rank %d)\n",
            tol, llama_rank, best_other,
            100 * llama_rank / r_true, 100 * best_other / r_true, r_true)
  end
end

main()
