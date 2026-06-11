using ElemCo
@testset "parse_orbstring" begin

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

@testset "SpinMatrix copy" begin
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

@testset "eri_2e4idx AO assembler" begin
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

@testset "eri_2e4idx_tri streaming triangular assembler" begin
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
