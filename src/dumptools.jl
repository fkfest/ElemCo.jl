"""
    DumpTools module

  Tools for manipulating FCIDump files.
"""
module DumpTools

using LinearAlgebra
using ..ElemCo.FciDump
using ..ElemCo.ECInfos
using ..ElemCo.FockFactory

export freeze_orbs_in_dump

"""
    freeze_orbs_in_dump(EC::ECInfo, freeze_orbs)

  Freeze orbitals in FCIDump file `EC.fd` according to an array or range `freeze_orbs`.
"""
function freeze_orbs_in_dump(EC::ECInfo, freeze_orbs)
  println("Freeze orbitals...")
  setup_space_fd!(EC)
  if EC.fd.uhf
    full_fock_a = gen_fock(EC, :α) 
    full_fock_b = gen_fock(EC, :β)
  else
    full_fock = gen_fock(EC)
  end
  space_save = save_space(EC)
  SP = EC.space
  freeze_occ = intersect(SP['o'], freeze_orbs)
  freeze_virt = intersect(SP['v'], freeze_orbs)
  ncore_orbs = freeze_nocc!(EC, freeze_occ)
  nfrozvirt = freeze_nvirt!(EC, 0, freeze_virt)
  nonfrozen = setdiff(SP[':'], freeze_occ, freeze_virt)
  nelec = headvar(EC.fd, "NELEC", Int) - 2*ncore_orbs
  norbs = headvar(EC.fd, "NORB", Int) - ncore_orbs - nfrozvirt
  @assert length(nonfrozen) == norbs
  if EC.fd.uhf
    core_fock_a = full_fock_a - gen_fock(EC, :α) 
    core_fock_b = full_fock_b - gen_fock(EC, :β)
    EC.fd.int0 += sum(diag(EC.fd.int1a)[freeze_occ]) + sum(diag(EC.fd.int1b)[freeze_occ]) + 
        0.5*(sum(diag(core_fock_a)[freeze_occ]) + sum(diag(core_fock_b)[freeze_occ]))

    EC.fd.int1a = EC.fd.int1a[nonfrozen,nonfrozen] + core_fock_a[nonfrozen,nonfrozen]
    EC.fd.int1b = EC.fd.int1b[nonfrozen,nonfrozen] + core_fock_b[nonfrozen,nonfrozen]
    EC.fd.int2aa = reorder_orbs_int2(EC.fd.int2aa, nonfrozen)
    EC.fd.int2bb = reorder_orbs_int2(EC.fd.int2bb, nonfrozen)
    EC.fd.int2ab = reorder_orbs_int2(EC.fd.int2ab, nonfrozen)
  else
    core_fock = full_fock - gen_fock(EC)
    EC.fd.int0 += 2*sum(diag(EC.fd.int1)[freeze_occ]) + sum(diag(core_fock)[freeze_occ]) 
    EC.fd.int1 = EC.fd.int1[nonfrozen,nonfrozen] + core_fock[nonfrozen,nonfrozen]
    EC.fd.int2 = reorder_orbs_int2(EC.fd.int2, nonfrozen)
  end
  modify_header!(EC.fd, norbs, nelec)
end

end #module