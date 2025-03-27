using ElemCo
using ElemCo.ECInfos

@testset "H2O Closed-Shell SVD-DC-CCSDT Test" begin
epsilon    =   1.e-6
EHF_test   = -75.6457645933
EDC_CCSDT_useT3_test = -0.33003948228084784 + EHF_test
EDC_CCSDT_test = -0.3300175166370924 + EHF_test
EDC_CCSDTp_test = -0.32981896554725615 + EHF_test

EDC_CCSDT_voXL_test = Dict(:combined  => -75.97578184022076,
                        :symcombined  => -75.97578210995636,
                        :triples      => -75.97579540955509,
                        :full         => -75.97578210995638)

EDC_CCSDTp_voXL_test = Dict(:combined => -75.97558419740024,
                        :symcombined  => -75.97558355886652,
                        :triples      => -75.97559061683168,
                        :full         => -75.97558355886655)

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

energies = @cc svd-dc-ccsdt 
@test abs(last_energy(energies)-EDC_CCSDT_test) < epsilon
@test abs(energies["SVD-DC-CCSDT+"]-EDC_CCSDTp_test) < epsilon

@set cc calc_t3_for_decomposition=true
energies = @cc svd-dc-ccsdt 
@test abs(last_energy(energies)-EDC_CCSDT_useT3_test) < epsilon

@set cc calc_t3_for_decomposition=false project_voXL=true
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
