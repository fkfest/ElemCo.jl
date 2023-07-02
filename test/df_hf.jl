@testset "DF-HF Closed-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -76.02253606201079
EMP2_test  =  -0.20694998731941067
EDCSD_test =  -0.22117576578925288

using ElemCo.MSystem
using ElemCo.DFHF
using ElemCo.DfDump

xyz="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")

ms = MSys(xyz,basis)

nelec = guess_nelec(ms)
norb = guess_norb(ms) 
occa = "-"*string(nelec÷2)
occb = "-"
EC = ECInfo()
mkpath(EC.scr)
EC.scr = mktempdir(EC.scr)
SP = EC.space
SP['o'], SP['v'], SP['O'], SP['V'] = get_occvirt(EC, occa, occb, norb, nelec)
SP[':'] = 1:norb

ϵ,cMO = dfhf(ms,EC,direct=true)

fcidump = "DF_HF_TEST.FCIDUMP"
dfdump(ms,EC,cMO,fcidump)

EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(EDCSD-EDCSD_test) < epsilon

rm(fcidump)

end
