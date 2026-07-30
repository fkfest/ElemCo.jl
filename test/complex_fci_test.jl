@testitem "complex_fci" tags=[:complex, :quick] begin
using Test
using ElemCo
using ElemCo.IntegralTools: transform_fcidump!
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex FCI/CIPHI" begin
epsilon = 1.e-6

geometry = "
  O      0.000000000    0.000000000   -0.130186067
  H1     0.000000000    1.489124508    1.033245507
  H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"6-31g", "jkfit"=>"vdz-jkfit", "mpfit"=>"vdz-mpfit")

# ============================================================
# RHF (closed-shell) tests
# ============================================================

@testset "Complex FCI - RHF" begin
  @dfhf
  # `@dfints`: this testset reuses `EC.fd` AFTER the run (to build a complex copy of it), so the
  # integrals have to be user-owned — a driver deletes the ones it creates for itself.
  @dfints
  energies_ref = @fci
  E_FCI_ref = energies_ref["FCI"]
  fd_real = EC.fd

  # --- Trivially complex (real integrals stored as ComplexF64) ---
  fd_c = FDump{ComplexF64,3}(fd_real)
  EC_c = ECInfo{ComplexF64}()
  EC_c.fd = fd_c
  energies_c = ElemCo.fcidriver(EC_c)
  @test abs(energies_c["FCI"] - E_FCI_ref) < epsilon

  # --- Diagonal phase rotation ---
  # FCI energy is invariant under any unitary orbital rotation.
  norb = headvar(fd_real, "NORB", Int)
  nelec = headvar(fd_real, "NELEC", Int)
  nocc = nelec ÷ 2
  β = 0.1
  phases = zeros(norb)
  phases[nocc+1:end] .= β
  U = diagm(exp.(im .* phases))

  fd_r = FDump{ComplexF64,3}(fd_real)
  transform_fcidump!(fd_r, SpinMatrix(conj(U)), SpinMatrix(U))
  EC_r = ECInfo{ComplexF64}()
  EC_r.fd = fd_r
  energies_r = ElemCo.fcidriver(EC_r)
  @test abs(energies_r["FCI"] - E_FCI_ref) < epsilon
end

@testset "Complex CIPHI - RHF" begin
  @dfhf
  @dfints                     # see above: `EC.fd` is reused after the run
  energies_ref = @ciphi
  E_CIPHI_ref = energies_ref["CIPHI"]
  fd_real = EC.fd

  norb = headvar(fd_real, "NORB", Int)
  nelec = headvar(fd_real, "NELEC", Int)
  nocc = nelec ÷ 2
  β = 0.1
  phases = zeros(norb)
  phases[nocc+1:end] .= β
  U = diagm(exp.(im .* phases))

  fd_r = FDump{ComplexF64,3}(fd_real)
  transform_fcidump!(fd_r, SpinMatrix(conj(U)), SpinMatrix(U))
  EC_r = ECInfo{ComplexF64}()
  EC_r.fd = fd_r
  energies_r = ElemCo.fcidriver(EC_r; ciphi=true)
  @test abs(energies_r["CIPHI"] - E_CIPHI_ref) < epsilon
end

# ============================================================
# UHF (open-shell) tests — H2O anion
# ============================================================

@testset "Complex FCI - UHF" begin
  @set wf charge=-1
  @dfuhf
  @dfints                     # see above: `EC.fd` is reused after the run
  energies_ref = @fci
  E_FCI_ref = energies_ref["FCI"]
  fd_real = EC.fd

  norb = headvar(fd_real, "NORB", Int)
  β = 0.1
  phases = zeros(norb)
  phases[2:end] .= β
  U = diagm(exp.(im .* phases))

  # Use 3-index FDump (compatible with ECInfo) and transform with unrestricted SpinMatrix.
  # Diagonal phase rotation preserves 3-index triangular symmetry.
  fd_r = FDump{ComplexF64,3}(fd_real)
  Uc = conj(U)
  transform_fcidump!(fd_r, SpinMatrix(Uc, copy(Uc)), SpinMatrix(copy(U), copy(U)))
  EC_r = ECInfo{ComplexF64}()
  EC_r.options.wf.charge = -1
  EC_r.fd = fd_r
  energies_r = ElemCo.fcidriver(EC_r)
  @test abs(energies_r["FCI"] - E_FCI_ref) < epsilon
end

@testset "Complex CIPHI - UHF" begin
  @set wf charge=-1
  @dfuhf
  @dfints                     # see above: `EC.fd` is reused after the run
  energies_ref = @ciphi
  E_CIPHI_ref = energies_ref["CIPHI"]
  fd_real = EC.fd

  norb = headvar(fd_real, "NORB", Int)
  β = 0.1
  phases = zeros(norb)
  phases[2:end] .= β
  U = diagm(exp.(im .* phases))

  fd_r = FDump{ComplexF64,3}(fd_real)
  Uc = conj(U)
  transform_fcidump!(fd_r, SpinMatrix(Uc, copy(Uc)), SpinMatrix(copy(U), copy(U)))
  EC_r = ECInfo{ComplexF64}()
  EC_r.options.wf.charge = -1
  EC_r.fd = fd_r
  energies_r = ElemCo.fcidriver(EC_r; ciphi=true)
  @test abs(energies_r["CIPHI"] - E_CIPHI_ref) < epsilon
end

end
end
