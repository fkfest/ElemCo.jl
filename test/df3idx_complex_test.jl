@testitem "df3idx_complex" tags=[:complex, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors: uppertriangular_index
using ElemCo.TensorTools: newmmap, flushmmap, closemmap
using LinearAlgebra

function int2_to_3idx_c(fd; thresh=1e-10)
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
      V[pq,rs] = fd.int2[r,p,uppertriangular_index(s,q)]
    end
  end
  evals, evecs = eigen(Hermitian(V))
  mask = evals .> thresh
  nL = count(mask)
  L = evecs[:,mask] * Diagonal(sqrt.(evals[mask]))
  return reshape(L, norb, norb, nL)
end

function setup_df3idx_ec_c(fd_orig, mmL_data)
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

@testset "DF-3IDX Complex" begin
  epsilon = 1e-6
  EHF_ref = -75.6457645933
  EMP2c_ref = -0.287815830908

  fcidump_path = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  fd_real = read_fcidump(fcidump_path)
  mmL_real = int2_to_3idx_c(fd_real)

  norb = headvar(fd_real, "NORB", Int)
  nelec = headvar(fd_real, "NELEC", Int)
  nocc = nelec ÷ 2

  # Diagonal phase rotation (uniform virtual phases)
  β = 0.1
  phases = zeros(norb)
  phases[nocc+1:end] .= β
  U = diagm(exp.(im .* phases))

  # Rotate mmL: mmL_c[:,:,L] = conj(U) * mmL_real[:,:,L] * transpose(U)
  nL = size(mmL_real, 3)
  mmL_c = zeros(ComplexF64, norb, norb, nL)
  Uc = conj(U)
  Ut = transpose(U)
  for L in 1:nL
    mmL_c[:,:,L] = Uc * mmL_real[:,:,L] * Ut
  end

  # Rotate int1: int1_c = conj(U) * int1_real * transpose(U)
  fd_c = FDump{ComplexF64,3}(fd_real)
  fd_c.int1 = Uc * fd_real.int1 * Ut

  @testset "Complex DF-HF with df3idx" begin
    EC = setup_df3idx_ec_c(fd_c, mmL_c)
    ehf = ElemCo.dfhf(EC)
    @test abs(ehf["HF"] - EHF_ref) < epsilon
  end

  @testset "Complex DF-UHF with df3idx" begin
    EC = setup_df3idx_ec_c(fd_c, mmL_c)
    euhf = ElemCo.dfuhf(EC)
    @test abs(euhf["HF"] - EHF_ref) < epsilon
  end

  @testset "Complex DF-MP2 with df3idx" begin
    EC = setup_df3idx_ec_c(fd_c, mmL_c)
    ElemCo.dfhf(EC)
    energies = ElemCo.dfccdriver(EC, "mp2")
    @test abs(energies["HF"] - EHF_ref) < epsilon
    @test abs(energies["MP2c"] - EMP2c_ref) < epsilon
  end
end
end
