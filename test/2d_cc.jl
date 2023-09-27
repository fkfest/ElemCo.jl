using ElemCo
@testset "Two-Determinant CCSD" begin
epsilon    =   1.e-6
td_ccsd_ref = -39.04463284087801

fcidump = joinpath(@__DIR__,"CH2.3B1.DZP.ROHF.FCIDUMP")

EC = ElemCo.ECInfo()
EHF, EMP2, ECC, W = ECdriver(EC, "2d-ccsd"; fcidump,occa="-2.1+1.3",occb="1.1+1.2+1.3")
@test abs(ECC+EHF-W-td_ccsd_ref) < epsilon

end
