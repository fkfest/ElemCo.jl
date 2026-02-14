using ElemCo
using ElemCo.ECInfos

@testset "H2O EOM Test" begin
epsilon    =   1.e-6
ω_CCSD_1_test  =  0.05715583935314445
ω_CCSD_2_test  =  0.08128887799115345
ω_DCSD_1_test  =  0.051543042500016324
ω_CCS_1_test   =  0.081167787111318836
ω_UCCSD_1_test  =  0.00869707336886641
ω_UDCSD_1_test  =  -0.0231164143063633
ω_UDCSD_2_test  =  0.0118629873623191
ω_RCCSD_1_test  =  -0.00669956844553506
ω_RCCSD_2_test  =  0.011997424137451085
ω_RDCSD_1_test  =  -0.00818467584083385
ω_RDCSD_2_test  =  0.015461981933628785
ω_UCCS_1_test  =  -0.0345771800271790
ω_UCCS_2_test  =  -0.0230285991986571
ω_RCCS_1_test  =  -0.0328163739213931

geometry = nothing
basis = nothing
fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

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
