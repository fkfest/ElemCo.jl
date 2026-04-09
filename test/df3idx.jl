using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors: uppertriangular_index
using ElemCo.TensorTools: newmmap, flushmmap, closemmap
using LinearAlgebra

"""
    int2_to_3idx(fd; thresh=1e-10)

Convert FciDump 4-index integrals to 3-index form via eigendecomposition.
Returns `mmL[p,q,L]` such that `int2[p,r,tri(q,s)] ≈ Σ_L mmL[p,q,L] * mmL[r,s,L]`.
"""
function int2_to_3idx(fd; thresh=1e-10)
  norb = headvar(fd, "NORB", Int)
  T = eltype(fd.int2)
  n2 = norb * norb
  V = zeros(T, n2, n2)
  for s=1:norb, r=1:norb, q=1:norb, p=1:norb
    pq = (q-1)*norb + p
    rs = (s-1)*norb + r
    if q <= s
      V[pq,rs] = fd.int2[p,r,uppertriangular_index(q,s)]
    else
      # (pq|rs) = (rs|pq), stored as int2[r,p,tri(s,q)]
      V[pq,rs] = fd.int2[r,p,uppertriangular_index(s,q)]
    end
  end
  evals, evecs = eigen(Hermitian(V))
  mask = evals .> thresh
  nL = count(mask)
  L = evecs[:,mask] * Diagonal(sqrt.(evals[mask]))
  return reshape(L, norb, norb, nL)
end

"""
    setup_df3idx_ec(fd_orig, mmL_data)

Set up an ECInfo with df3idx=true from a FciDump and pre-computed 3-index integrals.
"""
function setup_df3idx_ec(fd_orig, mmL_data)
  fd = deepcopy(fd_orig)
  T = eltype(mmL_data)
  norb = headvar(fd, "NORB", Int)
  nL = size(mmL_data, 3)
  fd.int2 = similar(fd.int2, 0, 0, 0)
  fd.df3idx = true
  EC = ECInfo{T}()
  EC.fd = fd
  mmLfile, mmL = newmmap(EC, "mmL", (norb, norb, nL); description="mmL")
  mmL .= mmL_data
  flushmmap(EC, mmL)
  closemmap(EC, mmLfile, mmL)
  return EC
end

@testset "DF-3IDX Real" begin
  epsilon = 1e-6
  EHF_ref = -75.6457645933
  EMP2c_ref = -0.287815830908

  fcidump_path = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  fd = read_fcidump(fcidump_path)
  mmL_data = int2_to_3idx(fd)

  @testset "DF-HF with df3idx" begin
    EC = setup_df3idx_ec(fd, mmL_data)
    ehf = ElemCo.dfhf(EC)
    @test abs(ehf["HF"] - EHF_ref) < epsilon
  end

  @testset "DF-UHF with df3idx" begin
    EC = setup_df3idx_ec(fd, mmL_data)
    euhf = ElemCo.dfuhf(EC)
    @test abs(euhf["HF"] - EHF_ref) < epsilon
  end

  @testset "DF-MP2 with df3idx" begin
    EC = setup_df3idx_ec(fd, mmL_data)
    ElemCo.dfhf(EC)
    energies = ElemCo.dfccdriver(EC, "mp2")
    @test abs(energies["HF"] - EHF_ref) < epsilon
    @test abs(energies["MP2c"] - EMP2c_ref) < epsilon
  end
end
