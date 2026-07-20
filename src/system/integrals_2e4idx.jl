# 2-electron 4-index integrals
# adapted from GaussianBasis.jl

"""
    eri_2e4idx_sph!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)

  Compute the two-electron four-index electron-repulsion integral block
  ``(ij|kl)`` (chemists' notation) for the spherical shells `i,j,k,l`.
  The result is stored in `out`.
"""
function eri_2e4idx_sph!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)
  cint2e_sph!(out, MVector(Cint(i-1),Cint(j-1),Cint(k-1),Cint(l-1)), basis.lib)
end

"""
    eri_2e4idx_cart!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)

  Compute the two-electron four-index electron-repulsion integral block
  ``(ij|kl)`` (chemists' notation) for the cartesian shells `i,j,k,l`.
  The result is stored in `out`.
"""
function eri_2e4idx_cart!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)
  cint2e_cart!(out, MVector(Cint(i-1),Cint(j-1),Cint(k-1),Cint(l-1)), basis.lib)
end

"""
    eri_2e4idx!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)

  Compute the two-electron four-index electron-repulsion integral block
  ``(ij|kl)`` (chemists' notation) for the shells `i,j,k,l`.
  The result is stored in `out`. Dispatches on the basis (spherical/cartesian).
"""
function eri_2e4idx!(out, i::Int, j::Int, k::Int, l::Int, basis::BasisSet)
  if is_cartesian(basis)
    eri_2e4idx_cart!(out, i, j, k, l, basis)
  else
    eri_2e4idx_sph!(out, i, j, k, l, basis)
  end
end

"""
    eri_2e4idx(ao_basis::BasisSet)

  Compute the full two-electron four-index electron-repulsion integral tensor
  ``(\\mu\\nu|\\rho\\sigma)`` in chemists' notation, i.e. `out[μ,ν,ρ,σ] = (μν|ρσ)`.

  The result is a dense `nao × nao × nao × nao` array. This is the non-density-fitted
  (exact) AO integral tensor. For storage in an [`FDump`](@ref) it has to be brought
  into physicists' notation (`<pq|rs> = (pr|qs)`); see the AO-dump assembler.
"""
function eri_2e4idx(ao_basis::BasisSet)
  nao = n_ao(ao_basis)
  out = zeros(nao, nao, nao, nao)
  eri_2e4idx!(out, ao_basis)
  return out
end

"""
    eri_2e4idx!(out, ao_basis::BasisSet)

  Compute the full two-electron four-index electron-repulsion integral tensor
  (chemists' notation) into `out` (`nao × nao × nao × nao`). Dispatches on the basis.
"""
function eri_2e4idx!(out, ao_basis::BasisSet)
  if is_cartesian(ao_basis)
    calc_2e4idx!(out, eri_2e4idx_cart!, ao_basis)
  else
    calc_2e4idx!(out, eri_2e4idx_sph!, ao_basis)
  end
  return out
end

"""
    calc_2e4idx!(out, callback::Function, ao_basis::BasisSet)

  Assemble the full four-index AO integral tensor `out[μ,ν,ρ,σ] = (μν|ρσ)`
  (chemists' notation) by looping over shell quartets and calling `callback`
  (one of [`eri_2e4idx_sph!`](@ref)/[`eri_2e4idx_cart!`](@ref)).

  Permutational symmetry is exploited at the shell-pair level
  (`(μν|··) = (νμ|··)` and `(··|ρσ) = (··|σρ)`, i.e. 4-fold reduction).
  The bra↔ket pair symmetry `(μν|ρσ) = (ρσ|μν)` is *not* exploited so that the
  parallel loop stays race-free: each task owns a distinct ket shell-pair `{K,L}`
  (keyed by `L = max(K,L)`), hence different tasks write disjoint `(ρ,σ)` blocks.
"""
function calc_2e4idx!(out, callback::Function, ao_basis::BasisSet)
  # Number of AOs per shell
  nao4sh = Int[n_ao(ash, ao_basis.cartesian) for ash in ao_basis]
  nao_max = maximum(nao4sh)
  nsh = length(nao4sh)

  # Offset list for each shell, used to map shell index to AO index
  ao_offset = cumsum(vcat(0, nao4sh))

  @threadsbuffer tbufs(Cdouble, nao_max^4) begin

  @sync for L in 1:nsh
    Threads.@spawn begin
      @inbounds begin
        buf = reshape_buf!(tbufs, length(tbufs))
        nl = nao4sh[L]
        Lblk = (1:nl) .+ ao_offset[L]
        for K in 1:L # Only upper triangle of the ket pair
          nk = nao4sh[K]
          Kblk = (1:nk) .+ ao_offset[K]
          for J in 1:nsh
            nj = nao4sh[J]
            Jblk = (1:nj) .+ ao_offset[J]
            for I in 1:J # Only upper triangle of the bra pair
              ni = nao4sh[I]
              Iblk = (1:ni) .+ ao_offset[I]

              # Call libcint: (IJ|KL) in chemists' notation
              callback(buf, I, J, K, L, ao_basis)
              vbuf = reshape_buf(buf, ni, nj, nk, nl)

              # (IJ|KL)
              out[Iblk, Jblk, Kblk, Lblk] = vbuf
              # (JI|KL)
              allocfree_permutedims!((@view out[Jblk, Iblk, Kblk, Lblk]), vbuf, (2,1,3,4))
              if K != L
                # (IJ|LK)
                allocfree_permutedims!((@view out[Iblk, Jblk, Lblk, Kblk]), vbuf, (1,2,4,3))
                # (JI|LK)
                allocfree_permutedims!((@view out[Jblk, Iblk, Lblk, Kblk]), vbuf, (2,1,4,3))
              end
            end
          end
        end
        reset!(tbufs)
      end #inbounds
    end #spawn
  end #sync
  end #threadsbuffer
  return out
end

"""
    eri_2e4idx_tri!(int2, ao_basis::BasisSet; target_length=100)

  Assemble the AO two-electron integrals **directly in the physicist-triangular
  layout** of an AO [`FDump`](@ref), dispatching on the basis (spherical/cartesian).
  See [`calc_2e4idx_tri!`](@ref).
"""
function eri_2e4idx_tri!(int2::AbstractArray{<:Number,3}, ao_basis::BasisSet; target_length::Int=100)
  if is_cartesian(ao_basis)
    calc_2e4idx_tri!(int2, eri_2e4idx_cart!, ao_basis; target_length)
  else
    calc_2e4idx_tri!(int2, eri_2e4idx_sph!, ao_basis; target_length)
  end
  return int2
end

"""
    calc_2e4idx_tri!(int2, callback::Function, ao_basis::BasisSet; target_length=100)

  Assemble the AO two-electron integrals **directly into the physicist-triangular
  layout** used by an AO [`FDump`](@ref), i.e.

      int2[p, q, tri(r,s)] = ⟨pq|rs⟩ = (pr|qs) = G[p,r,q,s]   (for r ≤ s),

  with `tri(r,s) = uppertriangular_index(r,s)` and `int2` of shape
  `(nao, nao, nao*(nao+1)÷2)`. The full dense `nao⁴` chemists' tensor is **never**
  materialized.

  The ket index `s` (physicist ket-2 = chemists' 4th index `σ`) is batched with a
  single-basis [`BasisBatcher`](@ref): each batch is a contiguous run of `s`-shells,
  hence owns a **contiguous block of `tri(r,s)` columns** of `int2` — written in
  I/O-friendly order and **race-free** (disjoint columns per batch). Within a batch
  the triangular symmetry is applied at the shell-pair level (`r`-shell ≤ `s`-shell);
  the bra indices `(p,q)` are full. The per-batch core [`eri_2e4idx_tri_batch!`](@ref)
  can likewise be used integral-direct (e.g. an AO Fock build) without an `int2`.
"""
function calc_2e4idx_tri!(int2::AbstractArray{T,3}, callback::Function, ao_basis::BasisSet;
                          target_length::Int=100) where {T}
  nao = n_ao(ao_basis)
  @assert size(int2) == (nao, nao, nao*(nao+1)÷2) "int2 has wrong shape for nao=$nao"
  bb = BasisBatcher(ao_basis, target_length)

  @threadsbuffer tbufs(Cdouble, buffer_size_4idx(bb)) begin
  @sync for batch in bb
    Threads.@spawn begin
      # contiguous tri-column block owned by this batch of s-shells (s ∈ batch.range)
      s_lo, s_hi = first(batch.range), last(batch.range)
      col_lo = s_lo*(s_lo-1)÷2 + 1        # uppertriangular_index(1, s_lo)
      col_hi = s_hi*(s_hi+1)÷2            # uppertriangular_index(s_hi, s_hi)
      slab = @view int2[:, :, col_lo:col_hi]
      eri_2e4idx_tri_batch!(slab, tbufs, callback, batch)
    end
  end
  end #threadsbuffer
  return int2
end

"""
    ket_shell_blocks(ao_basis::BasisSet; maxcols, target_length=100) -> Vector{Vector{BasisBatch}}

  Group the ket-2 (`s`) batches of a single-basis [`BasisBatcher`](@ref) into consecutive
  blocks of at most `maxcols` triangular `tri(r,s)` columns (each block keeps ≥ 1 batch, so
  a single oversized batch stands alone). Each block owns a contiguous, shell-aligned run of
  ket columns — a valid `σ`-blocking for the ± supermatrix store.
"""
function ket_shell_blocks(ao_basis::BasisSet; maxcols::Int, target_length::Int=100)
  bb = BasisBatcher(ao_basis, target_length)
  groups = Vector{BasisBatch}[]
  curcols = 0
  for batch in bb
    s_lo, s_hi = first(batch.range), last(batch.range)
    bcols = s_hi*(s_hi+1)÷2 - s_lo*(s_lo-1)÷2
    if isempty(groups) || (curcols + bcols > maxcols && curcols > 0)
      push!(groups, BasisBatch[]); curcols = 0
    end
    push!(groups[end], batch)
    curcols += bcols
  end
  return groups
end

"""
    calc_2e4idx_tri_blockwise!(consume!::Function, ao_basis::BasisSet, groups)

  Generate the physicist-triangular AO integrals **block by block**: for each group of
  `s`-batches (from [`ket_shell_blocks`](@ref)) the contiguous ket-column slab
  `slab[p, q, tri(r,s) − col_offset] = ⟨pq|rs⟩` is assembled in a reusable RAM buffer —
  batches within a block run in parallel over disjoint columns, exactly like
  [`calc_2e4idx_tri!`](@ref) — and handed to `consume!(J, slab)`. The full triangular
  array is never stored; e.g. the ± supermatrix store is folded directly from the slabs.
"""
function calc_2e4idx_tri_blockwise!(consume!::Function, ao_basis::BasisSet,
                                    groups::Vector{Vector{BasisBatch}})
  callback = is_cartesian(ao_basis) ? eri_2e4idx_cart! : eri_2e4idx_sph!
  nao = n_ao(ao_basis)
  maxblockcols = maximum(groups) do g
    s_lo = first(first(g).range); s_hi = last(last(g).range)
    s_hi*(s_hi+1)÷2 - s_lo*(s_lo-1)÷2
  end
  slab = zeros(Cdouble, nao, nao, maxblockcols)
  bb = first(first(groups)).bb
  @threadsbuffer tbufs(Cdouble, buffer_size_4idx(bb)) begin
  for (J, group) in enumerate(groups)
    s_lo = first(first(group).range); s_hi = last(last(group).range)
    col_lo = s_lo*(s_lo-1)÷2 + 1
    ncols = s_hi*(s_hi+1)÷2 - col_lo + 1
    @sync for batch in group
      Threads.@spawn begin
        b_lo = first(batch.range); b_hi = last(batch.range)
        bcol_lo = b_lo*(b_lo-1)÷2 + 1
        bcol_hi = b_hi*(b_hi+1)÷2
        sl = @view slab[:, :, (bcol_lo - col_lo + 1):(bcol_hi - col_lo + 1)]
        eri_2e4idx_tri_batch!(sl, tbufs, callback, batch)
      end
    end
    consume!(J, @view slab[:, :, 1:ncols])
  end
  end #threadsbuffer
  return
end

"""
    eri_2e4idx_tri_batch!(out, buffer, callback::Function, batch::BasisBatch)

  Fill one `s`-batch of the physicist-triangular AO integrals into the slab `out`,

      out[p, q, tri(r,s) - col_offset] = ⟨pq|rs⟩ = (pr|qs)   (for r ≤ s),

  where `batch` (from a single-basis [`BasisBatcher`](@ref)) supplies the `s`-shells
  and `col_offset = tri(1, s_lo) - 1` aligns the contiguous column block to `out`'s
  third dimension. The triangular symmetry is applied at the shell-pair level
  (`r`-shell ≤ `s`-shell, with the diagonal shell handled at the AO level); the bra
  shells `(p,q)` are full. `buffer` is a `Cdouble` (threads) buffer of size
  [`buffer_size_4idx`](@ref). This is the reusable core — pass an `int2` view to fill
  a memory-mapped dump, or a scratch slab to consume the block integral-direct.
"""
function eri_2e4idx_tri_batch!(out, buffer, callback::Function, batch::BasisBatch)
  bs = batch.bb.basis
  v = n4sh(batch, 1)                       # AO count per shell (single basis)
  off = bas_offset(batch, 1)               # AO offset per shell
  nsh = length(v)
  col_offset = (first(batch.range) - 1) * first(batch.range) ÷ 2   # tri(1, s_lo) - 1

  for Sb in batch.shrange                  # s-shell (ket-2)
    @inbounds begin
      buf = neuralyze(reshape_buf!(buffer, length(buffer)))
      nS = v[Sb]; Soff = off[Sb]
      for R in 1:Sb                        # r-shell (ket-1): r ≤ s ⇒ R ≤ S
        nR = v[R]; Roff = off[R]
        diagRS = (R == Sb)
        for Q in 1:nsh                     # q-shell (bra-2): full
          nQ = v[Q]; Qoff = off[Q]
          for P in 1:nsh                   # p-shell (bra-1): full
            nP = v[P]; Poff = off[P]
            # libcint: (P R | Q S) chemists' = (p r | q s), p∈P, r∈R, q∈Q, s∈S
            callback(buf, P, R, Q, Sb, bs)
            vbuf = reshape_buf(buf, nP, nR, nQ, nS)
            # scatter out[p, q, tri(r,s) - col_offset] = (pr|qs) for r ≤ s
            for sl in 1:nS
              s = sl + Soff
              tri_s = s*(s-1)÷2
              rmax = diagRS ? sl : nR      # within the diagonal shell enforce r ≤ s
              for rl in 1:rmax
                col = (rl + Roff) + tri_s - col_offset
                for ql in 1:nQ
                  q = ql + Qoff
                  @views out[(Poff+1):(Poff+nP), q, col] .= vbuf[:, rl, ql, sl]
                end
              end
            end
          end
        end
      end
      reset!(buffer)
    end #inbounds
  end
  return out
end
