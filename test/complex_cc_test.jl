@testitem "complex_cc" tags=[:complex, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex CC Ground State" begin
epsilon = 1.e-6

# Reference energies for H2O/cc-pVDZ
EHF_ref   = -75.6457645933
EMP2c_ref = -0.287815830908
EDCSD_ref = -0.328754956597 + EHF_ref
ECCSD_T_ref = -0.329259440500 + EHF_ref

fcidump_path = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
fd_real = read_fcidump(fcidump_path)

# Test 1: Trivially complex FCIDUMP (real integrals stored as ComplexF64)
# This validates the entire complex code pipeline with purely real data.
fd = FDump{ComplexF64,3}(fd_real)

EC = ECInfo{ComplexF64}()
EC.fd = fd
energies = ElemCo.ccdriver(EC, "dcsd"; fcidump="")

@test abs(energies["HF"] - EHF_ref) < epsilon
@test abs(energies["MP2c"] - EMP2c_ref) < epsilon
@test abs(last_energy(energies) - EDCSD_ref) < epsilon

# Test 2: Diagonal phase rotation with uniform occ/vir phases.
# All occupied orbital phases = 0, all virtual orbital phases = β.
# This produces truly complex integrals and amplitudes.
#
# Under this rotation, the CC amplitudes acquire phase factors that
# cancel exactly in the energy expression: each residual term's phase
# is determined only by external indices (uniform within occ/vir classes),
# so the converged energy is invariant.
β = 0.1
norb  = headvar(fd_real, "NORB", Int)
nelec = headvar(fd_real, "NELEC", Int)
nocc  = nelec ÷ 2  # closed-shell

phases = zeros(norb)
phases[nocc+1:end] .= β
U = diagm(exp.(im .* phases))

fd2 = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(fd2, SpinMatrix(conj(U)), SpinMatrix(U))

EC2 = ECInfo{ComplexF64}()
EC2.fd = fd2
energies2 = ElemCo.ccdriver(EC2, "dcsd"; fcidump="")

# HF energy is exactly invariant under diagonal phase rotation
@test abs(energies2["HF"] - EHF_ref) < epsilon

# Correlation energies are also invariant: amplitude phases cancel in energy
@test abs(energies2["MP2c"] - EMP2c_ref) < epsilon
@test abs(last_energy(energies2) - EDCSD_ref) < epsilon

# Test 3: CCSD(T) with complex rotation
EC3 = ECInfo{ComplexF64}()
EC3.fd = fd2
energies3 = ElemCo.ccdriver(EC3, "ccsd(t)"; fcidump="")
@test abs(energies3["CCSD(T)"] - ECCSD_T_ref) < epsilon

# Test 4: SVD-DC-CCSDT with complex rotation
# Use tight ampsvdtol to saturate the SVD basis (nX = nocc*nvirt),
# so the real/complex SVD bases are equivalent and the energy comparison
# is not limited by truncation-dependent differences.
# Run both real and complex with the same tight settings and compare directly.
EC4r = ECInfo()
EC4r.options.cc.ampsvdtol = 1e-5
energies4r = ElemCo.ccdriver(EC4r, "svd-dc-ccsdt"; fcidump=fcidump_path)
EC4c = ECInfo{ComplexF64}()
EC4c.fd = fd2
EC4c.options.cc.ampsvdtol = 1e-5
energies4c = ElemCo.ccdriver(EC4c, "svd-dc-ccsdt"; fcidump="")
@test abs(energies4c["SVD-DC-CCSDT"] - energies4r["SVD-DC-CCSDT"]) < epsilon

end
end
