using ElemCo
using ElemCo.ECInfos

@testset "H2O EOM Test" begin
epsilon    =   5.e-6
ω_CCSD_1_test  =  0.05718164988543591
ω_CCSD_2_test  =  0.08132557638663858
ω_DCSD_1_test  =  0.051551633211952434
ω_CCS_1_test   =  0.08116560792145354
ω_UCCSD_1_test  =  0.008680159161013397
ω_UDCSD_1_test  =  -0.02314639053493639
ω_UDCSD_2_test  =  0.011853449761153685
ω_RCCSD_1_test  =  -0.006697721715089196
ω_RCCSD_2_test  =  0.011991719800286415
ω_RDCSD_1_test  =  -0.008201151543078227
ω_RDCSD_2_test  =  0.015460362630355908
ω_UCCS_1_test  =  -0.03457768722123956
ω_UCCS_2_test  =  -0.02302860085064951
ω_RCCS_1_test  =  -0.03282201124598867

geometry = nothing
basis = nothing
fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

@set eom thr=1e-8

energies = @cc eom-ccsd begin
  @set eom nstates = 2
end
@test abs(energies["ω1"]-ω_CCSD_1_test) < epsilon
@test abs(energies["ω2"]-ω_CCSD_2_test) < epsilon

energies = @cc eom-dcsd
@test abs(energies["ω1"]-ω_DCSD_1_test) < epsilon

energies = @cc eom-ccs
@test abs(energies["ω1"]-ω_CCS_1_test) < epsilon

@set wf ms2 = 2

energies = @cc eom-uccsd
@test abs(energies["ω1"]-ω_UCCSD_1_test) < epsilon

energies = @cc eom-udcsd begin
  @set eom nstates = 2
end
@test abs(energies["ω1"]-ω_UDCSD_1_test) < epsilon
@test abs(energies["ω2"]-ω_UDCSD_2_test) < epsilon

energies = @cc eom-rccsd begin
  @set eom nstates = 2
end 
@test abs(energies["ω1"]-ω_RCCSD_1_test) < epsilon
@test abs(energies["ω2"]-ω_RCCSD_2_test) < epsilon

energies = @cc eom-rdcsd begin
  @set eom nstates = 2
end
@test abs(energies["ω1"]-ω_RDCSD_1_test) < epsilon
@test abs(energies["ω2"]-ω_RDCSD_2_test) < epsilon

energies = @cc eom-uccs begin
  @set eom nstates = 2
end
@test abs(energies["ω1"]-ω_UCCS_1_test) < epsilon
@test abs(energies["ω2"]-ω_UCCS_2_test) < epsilon
energies = @cc eom-rccs
@test abs(energies["ω1"]-ω_RCCS_1_test) < epsilon
end
