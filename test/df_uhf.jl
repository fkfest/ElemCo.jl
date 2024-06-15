using ElemCo

@testset "DF-HF Open-Shell Test" begin
epsilon    =  1.e-6
EUHF_test   =     -75.79199546194373
EUDCSD_test =     -0.1866586054908987 + EUHF_test
EUHF1_test  =     -75.63312357707606
EUCCSD1_test =    -0.1706009099216159 + EUHF1_test

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-rifit")
let
  @opt wf ms2=2
  EUHF = @dfuhf
  energies = @cc udcsd
  @test abs(last(EUHF)-EUHF_test) < epsilon
  @test abs(last(energies)-EUDCSD_test) < epsilon
end

let
  @opt wf charge=1 ms2=1
  @opt scf direct=true
  EUHF = @dfuhf 
  fcidump = "DF_UHF_TEST.FCIDUMP"
  @opt int fcidump=fcidump
  @dfints 
  energies = @cc uccsd fcidump=fcidump
  rm(fcidump)
  @test abs(last(EUHF)-EUHF1_test) < epsilon
  @test abs(last(energies)-EUCCSD1_test) < epsilon
end

end
