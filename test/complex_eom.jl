using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex EOM" begin
epsilon_eom = 5.e-6

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

# EOM-CCSD with complex rotation (2 states)
ω_CCSD_1_ref = 0.05718164988543591
ω_CCSD_2_ref = 0.08132557638663858
EC5 = ECInfo{ComplexF64}()
EC5.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC5.fd, SpinMatrix(conj(U)), SpinMatrix(U))
EC5.options.eom.thr = 1e-8
EC5.options.eom.nstates = 2
energies5 = ElemCo.ccdriver(EC5, "eom-ccsd"; fcidump="")
@test abs(energies5["ω1"] - ω_CCSD_1_ref) < epsilon_eom
@test abs(energies5["ω2"] - ω_CCSD_2_ref) < epsilon_eom

# EOM-DCSD with complex rotation
ω_DCSD_1_ref = 0.051551633211952434
EC6 = ECInfo{ComplexF64}()
EC6.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC6.fd, SpinMatrix(conj(U)), SpinMatrix(U))
EC6.options.eom.thr = 1e-8
energies6 = ElemCo.ccdriver(EC6, "eom-dcsd"; fcidump="")
@test abs(energies6["ω1"] - ω_DCSD_1_ref) < epsilon_eom

# EOM-UCCSD with complex rotation (separate alpha/beta phases)
ω_UCCSD_1_ref = 0.008680159161013397
EC7 = ECInfo{ComplexF64}()
EC7.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC7.fd, SpinMatrix(conj(U_a), conj(U_b)), SpinMatrix(U_a, U_b))
EC7.options.eom.thr = 1e-8
EC7.options.wf.ms2 = ms2
energies7 = ElemCo.ccdriver(EC7, "eom-uccsd"; fcidump="")
@test abs(energies7["ω1"] - ω_UCCSD_1_ref) < epsilon_eom

# EOM-UDCSD with complex rotation (2 states)
ω_UDCSD_1_ref = -0.02314639053493639
ω_UDCSD_2_ref = 0.011853449761153685
EC8 = ECInfo{ComplexF64}()
EC8.fd = FDump{ComplexF64,3}(fd_real)
transform_fcidump!(EC8.fd, SpinMatrix(conj(U_a), conj(U_b)), SpinMatrix(U_a, U_b))
EC8.options.eom.thr = 1e-8
EC8.options.wf.ms2 = ms2
EC8.options.eom.nstates = 2
energies8 = ElemCo.ccdriver(EC8, "eom-udcsd"; fcidump="")
@test abs(energies8["ω1"] - ω_UDCSD_1_ref) < epsilon_eom
@test abs(energies8["ω2"] - ω_UDCSD_2_ref) < epsilon_eom

end
