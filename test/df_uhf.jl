using ElemCo
using ElemCo.TrexioInterface

@testset "DF-HF Open-Shell Test" begin
epsilon    =  1.e-6
EUHF_test   =     -75.79199546194373
EUDCSD_test =     -0.1866586054908987 + EUHF_test
EUHF1_test  =     -75.63312357707606
EUCCSD1_test =    -0.1706009099216159 + EUHF1_test
μUHF_test   =       0.3943137411865568

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
  @test abs(EUHF["mu"]-μUHF_test) < epsilon
  @test abs(EUHF["DM"]-μUHF_test) < epsilon
  @opt cc properties=true
  @opt wf natorb="udcsd_natorb.h5"
  energies = @cc udcsd
  @test abs(last_energy(EUHF)-EUHF_test) < epsilon
  @test abs(last_energy(energies)-EUDCSD_test) < epsilon
  @test energies["mu"] > 0.0
  @test energies["DM"] > 0.0
  open_trexio(joinpath(EC.scr, "udcsd_natorb.h5"), "r") do io
    @test !ElemCo.TREXIO.trexio_has_rdm_1e(io)
    @test !ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
    @test !ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
    occa, occb = read_trexio_orbital_occupations(io, "mo")
    @test abs(sum(occa) - 6.0) < epsilon
    @test abs(sum(occb) - 4.0) < epsilon
  end
  open_trexio(ElemCo.Wavefunctions.dumpfile(EC, "w")[2], "r") do io
    @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
  end
  @opt cc properties=false
  @opt wf natorb=""
end

let
  @opt wf charge=1 ms2=1
  @opt scf direct=true
  EUHF = @dfuhf 
  fdump = "DF_UHF_TEST.FCIDUMP"
  @opt int fcidump=fdump
  @dfints
  @opt cc properties=true
  @opt wf natorb="uccsd_natorb.h5"
  energies = @cc uccsd fcidump=fdump
  rm(fdump)
  @test abs(last_energy(EUHF)-EUHF1_test) < epsilon
  @test abs(last_energy(energies)-EUCCSD1_test) < epsilon
  @test energies["mu"] > 0.0
  @test energies["DM"] > 0.0
  open_trexio(joinpath(EC.scr, "uccsd_natorb.h5"), "r") do io
    @test !ElemCo.TREXIO.trexio_has_rdm_1e(io)
    @test !ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
    @test !ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
    occa, occb = read_trexio_orbital_occupations(io, "mo")
    @test abs(sum(occa) - 5.0) < epsilon
    @test abs(sum(occb) - 4.0) < epsilon
  end
  open_trexio(ElemCo.Wavefunctions.dumpfile(EC, "w")[2], "r") do io
    @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
  end
  @opt cc properties=false
  @opt wf natorb=""
end

end
