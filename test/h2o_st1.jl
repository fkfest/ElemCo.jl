#using ..eCo
#using Test

@testset "H2O Closed-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.298014304953
EMp2_test  =  -0.069545740864
ECCSD_test =  -0.082041632192
EDCSD_test =  -0.082498102641

EC = ECInfo()
EC.fd = read_fcidump(joinpath(@__DIR__,"H2O_ST1.FCIDUMP"))
# create scratch directory
mkpath(EC.scr)
EC.scr = mktempdir(EC.scr)
norb = headvar(EC.fd, "NORB")
nelec = headvar(EC.fd, "NELEC")
occa = "-"
occb = "-"
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
# println("EHF: ", EHF)
# println("EHF_test: ", EHF_test)
@test abs(EHF-EHF_test) < epsilon

#calculate MP2
EMp2,T2 = calc_MP2(EC)
# println("EMp2: ", EMp2)
# println("EMp2_test: ", EMp2_test)
@test abs(EMp2-EMp2_test) < epsilon

#calculate CCSD
ecmethod = ECMethod("ccsd")
dc = (ecmethod.theory == "DC")
T1 = zeros(0)
if ecmethod.exclevel[1] == FullExc
    T1 = zeros(size(SP('v'),1),size(SP('o'),1))
end
ECCSD, T1, T2 = calc_cc(EC, T1, T2, dc)
@test abs(ECCSD-ECCSD_test) < epsilon

#calculate DCSD
ecmethod = ECMethod("dcsd")
dc = (ecmethod.theory == "DC")
T1 = zeros(0)
if ecmethod.exclevel[1] == FullExc
    T1 = zeros(size(SP('v'),1),size(SP('o'),1))
end
EMp2, T2 = calc_MP2(EC)
EDCSD, T1, T2 = calc_cc(EC, T1, T2, dc)
@test abs(EDCSD-EDCSD_test) < epsilon

end
