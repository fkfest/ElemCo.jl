using ..eCo
using Test

@testset "HF Test" begin
EC = ECInfo()
EC.fd = read_fcidump("be.fcidump")
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
EHF_test = -14.351880476202
epsilon = 1.e-12
@test abs(EHF-EHF_test) < epsilon
end
