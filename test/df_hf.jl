@testset "DF-HF Closed-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -76.02253606201079
EMP2_test  =  -0.20694998731941067
EDCSD_test =  -0.22117576578925288

try
  using ElemCo.MSystem
  using ElemCo.DFHF
  using ElemCo.DfDump
catch
  #using .MSystem
end

xyz="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")

EC = ECInfo(ms=MSys(xyz,basis))

setup!(EC)

ϵ,cMO = dfhf(EC,direct=true)

fcidump = "DF_HF_TEST.FCIDUMP"
dfdump(EC,cMO,fcidump)

EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(EDCSD-EDCSD_test) < epsilon

rm(fcidump)

end
