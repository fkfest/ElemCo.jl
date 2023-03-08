using ..eCo
using Test

@testset "H2O Open-Shell Test" begin
epsilon    =    1.e-6
EHF_test   =  -75.337282954481
EMP2_test  =  -0.207727619864
ECCSD_test =  -0.232320803220
EDCSD_test =  -0.243223819179

EC = ECInfo()
EC.fd = read_fcidump("test/H2O_CATION.FCIDUMP")
# create scratch directory
mkpath(EC.scr)
EC.scr = mktempdir(EC.scr)
norb = headvar(EC.fd, "NORB")
nelec = headvar(EC.fd, "NELEC")
occa = "1-4"
occb = "1-3"
EC.space['o'], EC.space['v'], EC.space['O'], EC.space['V'] = get_occvirt(EC, occa, occb, norb, nelec)
EC.space[':'] = 1:headvar(EC.fd,"NORB")

SP(sp::Char) = EC.space[sp]

closed_shell = (EC.space['o'] == EC.space['O'] && !EC.fd.uhf)

addname=""
if !closed_shell
addname = "U"
end

# calculate fock matrix 
if closed_shell
EC.fock,EC.ϵo,EC.ϵv = gen_fock(EC)
EC.fockb = EC.fock
EC.ϵob = EC.ϵo
EC.ϵvb = EC.ϵv
else
EC.fock,EC.ϵo,EC.ϵv = gen_fock(EC,SCα)
EC.fockb,EC.ϵob,EC.ϵvb = gen_fock(EC,SCβ)
end

# calculate HF energy
if closed_shell
EHF = sum(EC.ϵo) + sum(diag(integ1(EC.fd))[SP('o')]) + EC.fd.int0
else
EHF = 0.5*(sum(EC.ϵo)+sum(EC.ϵob) + sum(diag(integ1(EC.fd, SCα))[SP('o')]) + sum(diag(integ1(EC.fd, SCβ))[SP('O')])) + EC.fd.int0
end
@test abs(EHF-EHF_test) < epsilon

#calculate MP2
EMp2, T2a, T2b, T2ab = calc_UMP2(EC)
@test abs(EMp2-EMP2_test) < epsilon

# #calculate CCSD
# ecmethod = ECMethod("ccsd")
# dc = (ecmethod.theory == "DC")
# T1 = nothing
# if ecmethod.exclevel[1] == FullExc
#     T1 = zeros(size(SP('v'),1),size(SP('o'),1))
# end
# ECCSD = calc_cc!(EC, T1, T2, dc)
# @test abs(ECCSD-ECCSD_test) < epsilon

# #calculate DCSD
# ecmethod = ECMethod("dcsd")
# dc = (ecmethod.theory == "DC")
# T1 = nothing
# if ecmethod.exclevel[1] == FullExc
#     T1 = zeros(size(SP('v'),1),size(SP('o'),1))
# end
# EMp2, T2 = calc_MP2(EC)
# EDCSD = calc_cc!(EC, T1, T2, dc)
# @test abs(EDCSD-EDCSD_test) < epsilon

end
