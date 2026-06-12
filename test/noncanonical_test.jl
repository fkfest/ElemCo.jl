@testitem "noncanonical" tags=[:cc, :quick] begin
using ElemCo

@testset "Non-canonical (T) Test" begin
epsilon    =  1.e-6
ECCSD_T_test = -76.238992574422
EΛCCSD_T_test = -76.238950007049
EUCCSD_T_test = -76.080208726356
EΛUCCSD_T_test = -76.080133760103

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = "vdz" 

@dfhf begin
  @set scf maxit=1
end

energies = @cc ccsd(t)
@test abs(energies["CCSD(T)"]-ECCSD_T_test) < epsilon

energies = @cc λccsd(t)
@test abs(energies["ΛCCSD(T)"]-EΛCCSD_T_test) < epsilon

@set wf charge=-1

energies = @cc ccsd(t)
@test abs(energies["UCCSD(T)"]-EUCCSD_T_test) < epsilon

energies = @cc λccsd(t)
@test abs(energies["ΛUCCSD(T)"]-EΛUCCSD_T_test) < epsilon
end
end
