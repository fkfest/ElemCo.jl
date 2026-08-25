@testitem "parse_orbstring" tags=[:unit, :quick] begin
using ElemCo

orbs = "-3.1+-2.2+1.3"
orbsym = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
 2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
 3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
orbs_ref = [1,2,3,53,54,82]
@test ElemCo.parse_orbstring(orbs;orbsym) == orbs_ref

orbs = "-5+7-9"
orbsym = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1]
orbs_ref = [1,2,3,4,5,7,8,9]
@test ElemCo.parse_orbstring(orbs;orbsym) == orbs_ref
end

@testitem "SpinMatrix copy" tags=[:unit, :quick] begin
using ElemCo

restricted = ElemCo.SpinMatrix([1.0 0.0; 0.0 1.0])
restricted_copy = copy(restricted)
@test ElemCo.is_restricted(restricted_copy)
@test restricted_copy.α !== restricted.α

unrestricted = ElemCo.SpinMatrix([1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0])
unrestricted_copy = copy(unrestricted)
@test !ElemCo.is_restricted(unrestricted_copy)
@test unrestricted_copy.α !== unrestricted.α
@test unrestricted_copy.β !== unrestricted.β
end

@testitem "eri_2e4idx AO assembler" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx, eri_2e4idx_sph!

# brute-force reference: loop over ALL shell quartets, no symmetry reduction
function reference_4idx(bao)
  nao4sh = Int[n_ao(ash, bao.cartesian) for ash in bao]
  off = cumsum(vcat(0, nao4sh))
  nsh = length(nao4sh)
  nao = n_ao(bao)
  G = zeros(nao, nao, nao, nao)
  for I in 1:nsh, J in 1:nsh, K in 1:nsh, L in 1:nsh
    ni, nj, nk, nl = nao4sh[I], nao4sh[J], nao4sh[K], nao4sh[L]
    buf = zeros(ni*nj*nk*nl)
    eri_2e4idx_sph!(buf, I, J, K, L, bao)
    G[(1:ni).+off[I], (1:nj).+off[J], (1:nk).+off[K], (1:nl).+off[L]] = reshape(buf, ni, nj, nk, nl)
  end
  return G
end

geometry = "
  O   0.000000000   0.000000000  -0.130186067
  H1  0.000000000   1.489124508   1.033245507
  H2  0.000000000  -1.489124508   1.033245507"

for bname in ("sto-3g", "cc-pVDZ")
  ms = parse_geometry(geometry, Dict("ao"=>bname))
  bao = generate_basis(ms, "ao")
  G = eri_2e4idx(bao)
  # matches a naive full-loop assembly (correctness of the symmetry scatter)
  @test maximum(abs.(G .- reference_4idx(bao))) < 1e-12
  # 8-fold permutational symmetry of chemists' (μν|ρσ)
  @test maximum(abs.(G .- permutedims(G, (2,1,3,4)))) < 1e-12  # (μν|ρσ)=(νμ|ρσ)
  @test maximum(abs.(G .- permutedims(G, (1,2,4,3)))) < 1e-12  # (μν|ρσ)=(μν|σρ)
  @test maximum(abs.(G .- permutedims(G, (3,4,1,2)))) < 1e-12  # (μν|ρσ)=(ρσ|μν)
end
end

@testitem "eri_2e4idx_tri streaming triangular assembler" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx, eri_2e4idx_tri!
using ElemCo.QMTensors: uppertriangular_index

geometry = "
  O   0.000000000   0.000000000  -0.130186067
  H1  0.000000000   1.489124508   1.033245507
  H2  0.000000000  -1.489124508   1.033245507"

for bname in ("sto-3g", "cc-pVDZ")
  ms = parse_geometry(geometry, Dict("ao"=>bname))
  bao = generate_basis(ms, "ao")
  nao = n_ao(bao)
  G = eri_2e4idx(bao)                          # dense chemists' (μν|ρσ) reference
  # vary target_length to exercise batch boundaries (incl. tiny batches)
  for target_length in (100, 5, 1)
    int2 = zeros(nao, nao, nao*(nao+1)÷2)
    eri_2e4idx_tri!(int2, bao; target_length)  # batched streaming fill, no dense nao⁴ tensor
    # physicist-triangular layout: int2[p,q,tri(r,s)] == ⟨pq|rs⟩ == (pr|qs) == G[p,r,q,s], r ≤ s
    maxerr = 0.0
    for s in 1:nao, r in 1:s
      idx = uppertriangular_index(r, s)
      maxerr = max(maxerr, maximum(abs.(@view(int2[:, :, idx]) .- @view(G[:, r, :, s]))))
    end
    @test maxerr < 1e-12
  end
end
end

@testitem "transform_int2_Q rectangular + joint symmetry" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx
using ElemCo.IntegralTools: transform_int2_Q
using ElemCo.TensorTools: detri_int2
using ElemCo.QMTensors: uppertriangular_index
using LinearAlgebra, Random

geometry = "
  O   0.000000000   0.000000000  -0.130186067
  H1  0.000000000   1.489124508   1.033245507
  H2  0.000000000  -1.489124508   1.033245507"
bao = generate_basis(parse_geometry(geometry, Dict("ao"=>"sto-3g")), "ao")
nao = n_ao(bao)
G = eri_2e4idx(bao)                                   # chemists' (μν|ρσ)
# physicist-triangular packed int2[p,q,tri(r,s)] = <pq|rs> = G[p,r,q,s]  (encodes the joint
# (p↔q)+(r↔s) symmetry — the case that hides naive-contraction bugs)
int2 = zeros(nao, nao, nao*(nao+1)÷2)
for s in 1:nao, r in 1:s
  int2[:, :, uppertriangular_index(r, s)] = @view G[:, r, :, s]
end
vfull = detri_int2(int2, nao, 1:nao, 1:nao, 1:nao, 1:nao)   # full <p'q'|r's'>
# contract dim `d` of a 4-tensor with matrix X (X[idx,new]); allows a RECTANGULAR (nout<nin) X
reduce_idx(A, X, d) = (perm = (d, setdiff(1:4, d)...); Ap = permutedims(A, perm); n = size(Ap, 1);
    permutedims(reshape(X' * reshape(Ap, n, :), size(X, 2), size(Ap, 2), size(Ap, 3), size(Ap, 4)),
                invperm(collect(perm))))
Random.seed!(1)
for m in (nao, nao - 2)                               # square and rectangular
  Xa = randn(nao, m); Xb = randn(nao, m)              # asymmetric (Xa≠Xb) rectangular transforms
  out = transform_int2_Q(int2, Xa, Xb, Xa, Xb)        # v_{pq}^{rs}, p,r via Xa; q,s via Xb
  @test size(out) == (m, m, m, m)
  ref = reduce_idx(reduce_idx(reduce_idx(reduce_idx(vfull, Xa, 1), Xb, 2), Xa, 3), Xb, 4)
  @test maximum(abs.(out .- ref)) < 1e-10
end
end

@testitem "memory budget (available_memory / mem options)" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.Utils: available_memory, total_memory, free_memory, cgroup_memory_available, slurm_memory_limit

@test total_memory() > 0
@test 0 < free_memory() <= total_memory()
# auto estimate: positive, floored at 256 MiB, and bounded by a fraction of total (avail ≤ free ≤ total)
a = available_memory(fraction=0.6, gc=false)
@test a >= 256 * 2^20
@test a <= round(Int, 0.6 * total_memory())
# cgroup / SLURM helpers: return nothing (unconstrained) or a positive bound, never throw
@test cgroup_memory_available() === nothing || cgroup_memory_available() > 0
@test slurm_memory_limit() === nothing || slurm_memory_limit() > 0

EC = ElemCo.ECInfo()
@test EC.options.mem.budget == -1.0 && EC.options.mem.fraction == 0.8  # defaults: auto (≤0), 0.8
@test available_memory(EC) >= 256 * 2^20                               # auto path positive
@set mem budget=64.0                                                   # explicit override (GB) is exact
@test available_memory(EC) == round(Int, 64.0 * 2^30)
@set mem budget=-1.0                                                   # back to auto
@test available_memory(EC) >= 256 * 2^20
end

@testitem "int2 4-index transform (dense + triangular, single + multi block)" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.IntegralTools: transform_int2, transform_int2_Q, transform_int2!, transform_int2_Q!
using ElemCo.QMTensors: uppertriangular_index
using LinearAlgebra, Random

# reduce dim `d` of a 4-tensor by matrix X (X[idx,new]); allows rectangular nout<nin
reduce_idx(A, X, d) = (perm = (d, setdiff(1:4, d)...); Ap = permutedims(A, perm); n = size(Ap, 1);
    permutedims(reshape(X' * reshape(Ap, n, :), size(X, 2), size(Ap, 2), size(Ap, 3), size(Ap, 4)),
                invperm(collect(perm))))

Random.seed!(7)
for nin in (6, 9)
  # random input with the JOINT symmetry V[p',q',r',s'] = V[q',p',s',r'] (required for triangular storage)
  G = randn(nin,nin,nin,nin)
  V = G .+ permutedims(G, (2,1,4,3))
  int2 = zeros(nin, nin, nin*(nin+1)÷2)
  for s in 1:nin, r in 1:s
    int2[:, :, uppertriangular_index(r,s)] = @view V[:, :, r, s]
  end
  for nout in (nin, nin-2)                                  # square and rectangular
    Ca = randn(nin, nout); Cb = randn(nin, nout)
    # dense output, general Tl,Tl2,Tr,Tr2
    refQ = reduce_idx(reduce_idx(reduce_idx(reduce_idx(V, Ca,1), Cb,2), Ca,3), Cb,4)
    @test maximum(abs.(transform_int2_Q(int2, Ca, Cb, Ca, Cb) .- refQ)) < 1e-10
    outQb = zeros(nout,nout,nout,nout)                      # multi-block path (tiny budget → bsz=1)
    transform_int2_Q!(outQb, int2, Ca, Cb, Ca, Cb; membudget=1)
    @test maximum(abs.(outQb .- refQ)) < 1e-10
    # triangular output, same-spin Tl≡Tl2, Tr≡Tr2 — compare the r≤s packed columns to the dense ref
    refT = reduce_idx(reduce_idx(reduce_idx(reduce_idx(V, Ca,1), Ca,2), Ca,3), Ca,4)
    outT = transform_int2(int2, Ca, Ca, Ca, Ca)
    outTb = zeros(nout,nout,nout*(nout+1)÷2)
    transform_int2!(outTb, int2, Ca, Ca, Ca, Ca; membudget=1)
    eT = eTb = 0.0
    for s in 1:nout, r in 1:s
      idx = uppertriangular_index(r,s)
      eT  = max(eT,  maximum(abs.(outT[:,:,idx]  .- refT[:,:,r,s])))
      eTb = max(eTb, maximum(abs.(outTb[:,:,idx] .- refT[:,:,r,s])))
    end
    @test eT < 1e-10
    @test eTb < 1e-10
  end
  # degenerate reduce-to-empty (nout=0): must return empty output, not throw on maximum(empty)
  T0 = zeros(nin, 0)
  @test size(transform_int2(int2, T0, T0, T0, T0)) == (0, 0, 0)
  @test size(transform_int2_Q(int2, T0, T0, T0, T0)) == (0, 0, 0, 0)
end
end

@testitem "AO streaming Fock builder (ao_JK!/ao_J2K!)" tags=[:unit, :quick] begin
using ElemCo
using ElemCo.MSystems: parse_geometry
using ElemCo.Integrals: eri_2e4idx
using ElemCo.IntegralTools: ao_integrals
using ElemCo.TensorTools: detri_int2, mmap3idx
using ElemCo.FockFactory: ao_JK!, ao_J2K!
using ElemCo.PMStore: pm_to_joint!, open_pm_store, close_pm_store!
using ElemCo.ECInfos
using LinearAlgebra, Random

geometry = "
  O   0.000000000   0.000000000  -0.130186067
  H1  0.000000000   1.489124508   1.033245507
  H2  0.000000000  -1.489124508   1.033245507"

# dense physicist references: J[p,q]=Σ<pr|qs>Dj[r,s], K[p,q]=Σ<pr|sq>Dk[r,s]
jref(v, D) = (nao = size(v,1); reshape(reshape(permutedims(v,(1,3,2,4)), nao^2, nao^2)*vec(D), nao, nao))
kref(v, D) = (nao = size(v,1); reshape(reshape(permutedims(v,(1,4,2,3)), nao^2, nao^2)*vec(D), nao, nao))

Random.seed!(42)
for basis in ("sto-3g", "cc-pVDZ")
  EC = ECInfo{Float64}()
  EC.system = parse_geometry(geometry, Dict("ao"=>basis))
  ao_integrals(EC)                       # builds the ± supermatrix store
  # dense reference: reconstruct the jointly packed integrals from the store (`pm_to_joint!` is
  # exactly the inverse of the fold, so this compares the Fock builders against integrals that
  # never went through them)
  pm_to_joint!(EC)
  aofile, int2 = mmap3idx(EC, "ao_int2")
  nao = size(int2, 1)
  v = detri_int2(int2, nao, 1:nao, 1:nao, 1:nao, 1:nao)   # dense <pq|rs> reference
  close(aofile)
  pm = open_pm_store(EC)
  for sym in (true, false)   # HF densities are symmetric; biorthogonal ones need not be
    Dj = randn(nao,nao); Dk = randn(nao,nao); Da = randn(nao,nao); Db = randn(nao,nao)
    if sym
      Dj += Dj'; Dk += Dk'; Da += Da'; Db += Db'
    end
    # fused Coulomb+exchange, no dense nao⁴ tensor formed
    J = zeros(nao,nao); K = zeros(nao,nao); ao_JK!(J, K, pm, Dj, Dk)
    @test maximum(abs.(J .- jref(v,Dj))) < 1e-11
    @test maximum(abs.(K .- kref(v,Dk))) < 1e-11
    # UHF: shared Coulomb from the total density + two same-spin exchanges in one pass
    Dt = Da .+ Db
    J2 = zeros(nao,nao); Ka = zeros(nao,nao); Kb = zeros(nao,nao)
    ao_J2K!(J2, Ka, Kb, pm, Dt, Da, Db)
    @test maximum(abs.(J2 .- jref(v,Dt))) < 1e-11
    @test maximum(abs.(Ka .- kref(v,Da))) < 1e-11
    @test maximum(abs.(Kb .- kref(v,Db))) < 1e-11
  end
  close_pm_store!(EC, pm)
end
end

# MIO positional (offset-addressed) I/O: mioheadersize gives the data offset from the mio header
# structure; miopread! reads an exact byte range at an absolute offset (matches the mmapped data);
# mioprefetch is an advisory best-effort no-op-safe hint. These back the out-of-core pm_half_trans reads.
@testitem "MIO positional IO (mioheadersize / miopread! / mioprefetch)" tags=[:unit, :quick] begin
  using ElemCo.MIO: mionewmmap, miommap, mioclosemmap, mioheadersize, miopread!, mioprefetch
  for T in (Float64, ComplexF64)
    dims = (4, 3, 5); ntot = prod(dims); sz = sizeof(T)
    data = randn(T, dims)
    fname = tempname()
    io, arr = mionewmmap(fname, dims, T)
    arr .= data; mioclosemmap(io, arr)
    io2, _ = miommap(fname, Val(3), T)
    hdr = mioheadersize(io2)
    @test hdr == (3 + length(dims)) * sizeof(Int)          # type code + narray + ndim + dims
    flat = vec(data)
    # read a middle element range [7:19] positionally and compare (exact — same bits written/read)
    i0, i1 = 7, 19; nel = i1 - i0 + 1
    buf = Vector{T}(undef, nel)
    GC.@preserve buf miopread!(io2, pointer(buf), nel*sz, hdr + (i0-1)*sz)
    @test buf == flat[i0:i1]
    # the whole array read positionally == the flat data
    buf2 = Vector{T}(undef, ntot)
    GC.@preserve buf2 miopread!(io2, pointer(buf2), ntot*sz, hdr)
    @test buf2 == flat
    # short read at the tail must error (asks for one element past the end)
    @test_throws Exception (GC.@preserve buf miopread!(io2, pointer(buf), 2*sz, hdr + (ntot-1)*sz))
    @test mioprefetch(io2, hdr, ntot*sz) === nothing       # advisory hint: best-effort, never throws
    close(io2)
    rm(fname; force=true)
  end
end
