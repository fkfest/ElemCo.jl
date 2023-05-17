module DFMCSCF
using LinearAlgebra, TensorOperations, Printf
using ..ECInfos
using ..ECInts
using ..MSystem
using ..DIIS
using ..TensorTools
using ..DFHF

"""
calc density matrix of active electrons
D1[p,q] = <E_pq> = <a†_p a_q>
D2[p,q,r,s] = <E_pq,rs> = <E_pq E_rs - δ_qr E_ps> = <a†_p a†_r a_s a_q>

return as a tuple: D1, D2

"""
function denMatCreate(EC::ECinfo)
    SP = EC.space
    nact = length(SP['o'])- length(SP['O'])
    D1 = Matrix(I, nact, nact)
    @tensoropt D2[p,q,r,s] := D1[p,q]*D1[r,s] - 0.5*D1[p,q]*D1[r,s]
    return D1, D2
end

function dffockAS(EC,cMO,D1)
    occ2 = intersect(EC.space['o'],EC.space['O']) # to be modified
    occ1o = setdiff(EC.space['o'],occ2)
    occ1O = setdiff(EC.space['O'],occ2)
    CMO2 = cMO[:,occ2]
    pqL = load(EC,"munuL")
    hsmall = load(EC,"hsmall")
    @tensoropt pjL[p,j,L] := pqL[p,q,L] * CMO2[q,j]

    @tensoropt L[L] := pjL[p,j,L] * CMO2[p,j]
    @tensoropt fockClosed[p,q] := hsmall[p,q] - pjL[p,j,L]*pjL[q,j,L]
    @tensoropt fockClosed[p,q] += 2.0*L[L]*pqL[p,q,L]
    fock =  deepcopy(fockClosed)
    @tensoropt fock[p,q] += D1[t,u]*()

end


function dfmcscf(ms::MySys, EC::ECInfo; direct = false, guess = GUESS_SAD)
    Enuc = generate_integrals(ms, EC; save3idx=!direct)
    cMO = guess_orb(ms,EC,guess)

end

end #module