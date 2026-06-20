@testitem "df_hf" tags=[:df, :quick] begin
using ElemCo
using ElemCo.TrexioInterface

@testset "DF-HF Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_test   =      -76.02145513971418
EMP2_test  =      -0.204723138509385 + EHF_test
EDCSD_test =      -0.219150244853825 + EHF_test
ESVDDCSD_test =   -0.219409334393 + EHF_test
ESVDDCSD_ft_test =-0.220499791372 + EHF_test
EUHF_test  =      -75.79199546193901
μHF_test   =        2.103366881397954
μMP2_test  =        2.0801580304766434

orbital_printout_test = "Opening dump file wf.h5 for reading ...\nFetching orbitals ...\nRead DF-HF molecular orbitals from TREXIO file\n4:5 orbitals from DF-HF\n4:  0.788(O[1]1p{z})  0.353(H1[2]1s)  0.353(H2[3]1s) -0.290(O[1]2s) -0.170(O[1]3s) \n5:  0.922(O[1]1p{x}) \n"

xyz="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-mpfit")

EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz,basis))

@set scf direct=true
hf_energies = @dfhf
@test abs(hf_energies["HF"]-EHF_test) < epsilon
@test abs(hf_energies["mu"]-μHF_test) < epsilon
@test abs(hf_energies["DM"]-μHF_test) < epsilon
@test abs(hf_energies["DMZ"]-hf_energies["muz"]) < epsilon
@test last(keys(hf_energies)) == "E"
# store orbital printout in a string
original_stdout = stdout
(rd, wr) = redirect_stdout();
@show_orbs 4:5
redirect_stdout(original_stdout)
close(wr)
orbital_printout = read(rd, String)
close(rd)
println(orbital_printout)
@test orbital_printout == orbital_printout_test 

energies = @dfmp2
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test !ElemCo.file_exists(EC, "T_vvoo")

dfmp2_natorb = "dfmp2_natorb.h5"
dfmp2_store = "dfmp2_store.h5"
@set cc properties=true
@set wf store=dfmp2_store
@set wf natorb=dfmp2_natorb
energies = @dfmp2
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(energies["mu"]-μMP2_test) < epsilon
@test abs(energies["DM"]-μMP2_test) < epsilon
@test haskey(energies, "DMX")
@test haskey(energies, "DMY")
@test haskey(energies, "DMZ")
@test last(keys(energies)) == "E"
open_trexio(joinpath(EC.scr, dfmp2_natorb), "r") do io
     @test !ElemCo.TREXIO.trexio_has_rdm_1e(io)
     occa, occb = read_trexio_orbital_occupations(io, "mo")
     @test isempty(occb)
     @test abs(sum(occa) - 10.0) < epsilon
end
open_trexio(ElemCo.Wavefunctions.dumpfile(EC, "w")[2], "r") do io
     @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
end
@test !ElemCo.file_exists(EC, "T_vvoo")
@set cc properties=false
@set wf store=""
@set wf natorb=""

fdump = "DF_HF_TEST.FCIDUMP"
@set int fcidump=fdump
@dfints

dcsd_natorb = "dcsd_natorb.h5"
@set cc properties=true
@set wf natorb=dcsd_natorb
energies = ElemCo.ccdriver(EC, "dcsd"; fcidump=fdump)
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(energies["DCSD"]-EDCSD_test) < epsilon
@test energies["mu"] > 0.0
@test energies["DM"] > 0.0
@test last(keys(energies)) == "E"
open_trexio(joinpath(EC.scr, dcsd_natorb), "r") do io
     @test !ElemCo.TREXIO.trexio_has_rdm_1e(io)
     occa, occb = read_trexio_orbital_occupations(io, "mo")
     @test isempty(occb)
     @test abs(sum(occa) - 10.0) < epsilon
end
open_trexio(ElemCo.Wavefunctions.dumpfile(EC, "w")[2], "r") do io
     @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
end
@set cc properties=false
@set wf natorb=""

rm(fdump)

energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_test) < epsilon
@set cc use_full_t2=true
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft_test) < epsilon

# Test MO-first half-transform route (triggered when norbs ≤ nao/2)
@set cc use_full_t2=false
@set int fcidump=""
@set wf ms2=0 freeze_nvirt=12
EC.fd = ElemCo.FciDumps.FDump{Float64,3}()
@dfhf
EDCSD_fv_test = -76.12720216862047
energies = @cc dcsd
@test abs(energies["DCSD"]-EDCSD_fv_test) < epsilon

@set scf direct=false
@set wf ms2=2 freeze_nvirt=0
EUHF = @dfuhf
@test abs(EUHF["HF"]-EUHF_test) < epsilon
end
end
