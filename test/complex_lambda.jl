using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex Lambda" begin
epsilon = 1.e-6

fcidump_path = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
fd_real = read_fcidump(fcidump_path)

# Setup closed-shell complex rotation
β = 0.1
norb  = headvar(fd_real, "NORB", Int)
nelec = headvar(fd_real, "NELEC", Int)
nocc  = nelec ÷ 2

phases = zeros(norb)
phases[nocc+1:end] .= β
U = diagm(exp.(im .* phases))

# Setup unrestricted complex rotation (ms2=2, separate alpha/beta)
ms2 = 2
nocc_a = (nelec + ms2) ÷ 2
nocc_b = (nelec - ms2) ÷ 2
phases_a = zeros(norb)
phases_a[nocc_a+1:end] .= β
U_a = diagm(exp.(im .* phases_a))
phases_b = zeros(norb)
phases_b[nocc_b+1:end] .= β
U_b = diagm(exp.(im .* phases_b))

# Reference energies
EHF_ref = -75.6457645933
EHF_triplet_ref = -75.62407982361415

# ΛCCSD(T) with complex rotation (use_pm_kext=true)
EΛCCSD_T_ref = -0.326915143863 + EHF_ref
EC1 = ECInfo{ComplexF64}()
EC1.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC1.fd, SpinMatrix(conj(U)), SpinMatrix(U))
EC1.options.cc.use_pm_kext = true
energies1 = ElemCo.ccdriver(EC1, "λccsd(t)"; fcidump="")
@test abs(energies1["ΛCCSD(T)"] - EΛCCSD_T_ref) < epsilon

# ΛUCCSD(T) with complex rotation (T0 variant, fock_diag_thr=-1.0)
EΛUCCSD_T0_ref = -0.2903721324779 + EHF_triplet_ref
EC2 = ECInfo{ComplexF64}()
EC2.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC2.fd, SpinMatrix(conj(U_a), conj(U_b)), SpinMatrix(U_a, U_b))
EC2.options.wf.ms2 = ms2
EC2.options.cc.fock_diag_thr = -1.0
energies2 = ElemCo.ccdriver(EC2, "λuccsd(t)"; fcidump="")
@test abs(last_energy(energies2) - EΛUCCSD_T0_ref) < epsilon

# ΛUCCSD(T) with complex rotation (T variant, default)
EΛUCCSD_T_ref = -0.290495749316 + EHF_triplet_ref
EC3 = ECInfo{ComplexF64}()
EC3.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC3.fd, SpinMatrix(conj(U_a), conj(U_b)), SpinMatrix(U_a, U_b))
EC3.options.wf.ms2 = ms2
energies3 = ElemCo.ccdriver(EC3, "λuccsd(t)"; fcidump="")
@test abs(last_energy(energies3) - EΛUCCSD_T_ref) < epsilon

end
