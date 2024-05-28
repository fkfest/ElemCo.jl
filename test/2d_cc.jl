using ElemCo

@testset "Two-Determinant CCSD CAS" begin

epsilon    =   1.e-6
td_ccsd_ref = -39.044570270428
frt_ccsd_ref = -39.043778623741794

fcidump = joinpath(@__DIR__,"CH2.3B1.DZP.ROHF.FCIDUMP")

EC = ElemCo.ECInfo()
energies = ElemCo.ccdriver(EC, "2d-ccsd"; fcidump, occa="-2.1+1.3", occb="1.1+1.2+1.3")
println(keys(energies))
@test abs(energies[:TRIP2D_UCCSD]-td_ccsd_ref) < epsilon

energies = @cc frt-ccsd occa="-2.1+1.3" occb="1.1+1.2+1.3"
@test abs(energies[:FRT_UCCSD]-frt_ccsd_ref) < epsilon

end

@testset "Two-Determinant CCSD IAS" begin

epsilon    =   1.e-6
td_dcsd_ref = -113.824157087033

fcidump = joinpath(@__DIR__,"CH2O.3A1.VDZ.ROHF.FCIDUMP")

EC = ElemCo.ECInfo()
@opt cc nomp2=1 maxit=200
energies = ElemCo.ccdriver(EC, "2d-dcsd"; fcidump, occa = "-3.1+1.2+-2.3", occb = "-3.1+2.2+-2.3")
@test abs(energies[:SING2D_UDCSD]-td_dcsd_ref) < epsilon

end
