@testitem "complex_bohf" tags=[:complex, :quick] begin
using Test
using ElemCo
using ElemCo.IntegralTools: transform_fcidump!
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex BOHF/BOUHF" begin
epsilon = 1.e-6

fcidump = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
geometry = nothing
# ============================================================
# Reference real calculations
# ============================================================
@opt scf guess=:hcore temperature_guess=1e9
energies_real_bohf = @bohf
E_BOHF_ref = energies_real_bohf["HF"]

energies_real_bouhf = @bouhf
E_BOUHF_ref = energies_real_bouhf["UHF"]

# Create complex integrals via diagonal phase rotation
fd_real = EC.fd
norb = headvar(fd_real, "NORB", Int)
nelec = headvar(fd_real, "NELEC", Int)
nocc = nelec ÷ 2
β = 0.1
phases = zeros(norb)
phases[nocc+1:end] .= β
U = diagm(exp.(im .* phases))
fd_c = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(fd_c, SpinMatrix(conj(U)), SpinMatrix(U))

# ============================================================
# Complex BOHF
# ============================================================
@testset "Complex BOHF" begin
  EC_c = ECInfo{ComplexF64}()
  EC_c.fd = fd_c
  EC_c.options.scf.guess = :hcore
  EC_c.options.scf.temperature_guess = 1e9
  energies_c = ElemCo.BOHF.bohf(EC_c)
  @test abs(energies_c["HF"] - E_BOHF_ref) < epsilon
end

# ============================================================
# Complex BOUHF
# ============================================================
@testset "Complex BOUHF" begin
  EC_c = ECInfo{ComplexF64}()
  EC_c.fd = fd_c
  EC_c.options.scf.guess = :hcore
  EC_c.options.scf.temperature_guess = 1e9
  energies_c = ElemCo.BOHF.bouhf(EC_c)
  @test abs(energies_c["UHF"] - E_BOUHF_ref) < epsilon
end

end
end
