using ElemCo
using ElemCo.ECInfos

@testset "H2O Closed-Shell SVD-DC-CCSDT Test" begin
epsilon    =   1.e-6
EHF_test   = -75.6457645933
EDC_CCSDT_useT3_test = -0.330394818949 + EHF_test
EDC_CCSDT_test = -0.330199442786 + EHF_test
EDC_CCSDTp_test = -0.329890770108 + EHF_test

EDC_CCSDT_voXL_test = Dict(:combined  => -75.975963232268,
                        :symcombined  => -75.975963992872,
                        :triples      => -75.975968047548,
                        :full         => -75.975964036106)

EDC_CCSDTp_voXL_test = Dict(:combined => -75.975657017185,
                        :symcombined  => -75.975655349732,
                        :triples      => -75.975657802073,
                        :full         => -75.975655363428)
geometry = basis = nothing
fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

energies = @cc svd-dc-ccsdt 
@test abs(last_energy(energies)-EDC_CCSDT_test) < epsilon
@test abs(energies["SVD-DC-CCSDT+"]-EDC_CCSDTp_test) < epsilon

energies = @cc svd-dc-ccsdt begin 
  @set cc calc_t3_for_decomposition=true
end
@test abs(last_energy(energies)-EDC_CCSDT_useT3_test) < epsilon

@set cc project_voXL=true
energies = @cc svd-dc-ccsdt
@test abs(last_energy(energies)-EDC_CCSDT_voXL_test[:combined]) < epsilon
@test abs(energies["SVD-DC-CCSDT+"]-EDC_CCSDTp_voXL_test[:combined]) < epsilon

for sp in [:symcombined, :triples, :full]
  @set cc space4voXL=sp
  energies = @cc svd-dc-ccsdt 
  @test abs(last_energy(energies)-EDC_CCSDT_voXL_test[sp]) < epsilon
  @test abs(energies["SVD-DC-CCSDT+"]-EDC_CCSDTp_voXL_test[sp]) < epsilon
end

end
