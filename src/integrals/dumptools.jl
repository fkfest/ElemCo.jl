"""
    DumpTools module

  Tools for manipulating FCIDump files.
"""
module DumpTools

using LinearAlgebra
using ..ElemCo.FciDumps
using ..ElemCo.ECInfos
using ..ElemCo.FockFactory

export freeze_orbs_in_dump

"""
    freeze_orbs_in_dump(EC::ECInfo, freeze_orbs; int2alloc=nothing)

  Freeze orbitals in FCIDump file `EC.fd` according to an array or range `freeze_orbs`.

  By default the reduced (active) 2-e integrals are reordered into a new in-memory array. Pass
  `int2alloc(key, dims) -> array` (e.g. an mmap allocator; `key` ∈ `"int2"`, `"int2aa"`,
  `"int2bb"`, `"int2ab"`) to write the reduced integrals straight onto a memory-mapped scratch
  file instead — this avoids materializing the active block in memory and any extra copy.
"""
function freeze_orbs_in_dump(EC::ECInfo, freeze_orbs; int2alloc=nothing)
  reord(key, int2) = isnothing(int2alloc) ? reorder_orbs_int2(int2, nonfrozen) :
                     reorder_orbs_int2(int2, nonfrozen; alloc=dims->int2alloc(key, dims))
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
    EC.fd.int2aa = reord("int2aa", EC.fd.int2aa)
    EC.fd.int2bb = reord("int2bb", EC.fd.int2bb)
    EC.fd.int2ab = reord("int2ab", EC.fd.int2ab)
  else
    core_fock = full_fock - gen_fock(EC)
    EC.fd.int0 += 2*sum(diag(EC.fd.int1)[freeze_occ]) + sum(diag(core_fock)[freeze_occ])
    EC.fd.int1 = EC.fd.int1[nonfrozen,nonfrozen] + core_fock[nonfrozen,nonfrozen]
    EC.fd.int2 = reord("int2", EC.fd.int2)
  end
  modify_header!(EC.fd, norbs, nelec)
end

"""
    freeze_orbs_in_dump(EC::ECInfo, freeze_orbs::AbstractString)

  Freeze orbitals in FCIDump file `EC.fd` according to a string of indices `freeze_orbs`.

  `freeze_orbs` is a string of indices in format `"1-4+6-8"` where `-` denotes a range and `+` 
  denotes a list of indices or `"1:4;6:8"` where `:` denotes a range and `;` denotes a list of indices 
  (== `[1,2,3,4,6,7,8]`).
"""
function freeze_orbs_in_dump(EC::ECInfo, freeze_orbs::AbstractString)
  freeze_orbs = parse_orbstring(freeze_orbs)
  freeze_orbs_in_dump(EC, freeze_orbs)
end
end #module