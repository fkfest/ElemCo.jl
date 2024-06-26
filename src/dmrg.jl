"""
    DMRG

Density Matrix Renormalization Group (DMRG) calculations
using `ITensors.jl` package.
"""
module DMRG
using ITensors
using Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.FciDumps

export calc_dmrg

"""
    gen_hamiltonian(EC::ECInfo)

  Generate the Hamiltonian tensor for the DMRG calculation.
"""
function gen_hamiltonian(EC::ECInfo, atol=1e-15)
  ham = OpSum()
  @assert !EC.fd.uhf "Only restricted HF is supported for DMRG calculations."
  # 1. Add E_core
  add!(ham, EC.fd.int0, "Id", 1)
  # 2. Add 1-electron part
  norb = n_orbs(EC)
  int1 = integ1(EC.fd)
  for p in 1:norb, q in 1:norb
    if norm(int1[p, q]) > atol
      add!(ham, int1[p, q], "c†↑", p, "c↑", q)
      add!(ham, int1[p, q], "c†↓", p, "c↓", q)
    end
  end
  # 3. Add 2-electron part
  int2 = integ2(EC.fd)
  if ndims(int2) == 4
    for p in 1:norb, q in 1:norb, r in 1:norb, s in 1:norb
      if norm(int2[p, q, r, s]) > atol
        add!(ham, 0.5*int2[p, q, r, s], "c†↑", p, "c†↑", q, "c↑", s, "c↑", r)
        add!(ham, 0.5*int2[p, q, r, s], "c†↓", p, "c†↓", q, "c↓", s, "c↓", r)
        add!(ham, 0.5*int2[p, q, r, s], "c†↓", p, "c†↑", q, "c↑", s, "c↓", r)
        add!(ham, 0.5*int2[p, q, r, s], "c†↑", p, "c†↓", q, "c↓", s, "c↑", r)
      end
    end
  else
    # last two indices of integrals are stored as upper triangular 
    for s in 1:norb, r in 1:s, q in 1:norb, p in 1:norb
      rs = uppertriangular_index(r, s)
      if norm(int2[p, q, rs]) > atol
        add!(ham, 0.5*int2[p, q, rs], "c†↑", p, "c†↑", q, "c↑", s, "c↑", r)
        add!(ham, 0.5*int2[p, q, rs], "c†↓", p, "c†↓", q, "c↓", s, "c↓", r)
        add!(ham, 0.5*int2[p, q, rs], "c†↓", p, "c†↑", q, "c↑", s, "c↓", r)
        add!(ham, 0.5*int2[p, q, rs], "c†↑", p, "c†↓", q, "c↓", s, "c↑", r)
        if r != s
          add!(ham, 0.5*int2[q, p, rs], "c†↑", p, "c†↑", q, "c↑", r, "c↑", s)
          add!(ham, 0.5*int2[q, p, rs], "c†↓", p, "c†↓", q, "c↓", r, "c↓", s)
          add!(ham, 0.5*int2[q, p, rs], "c†↓", p, "c†↑", q, "c↑", r, "c↓", s)
          add!(ham, 0.5*int2[q, p, rs], "c†↑", p, "c†↓", q, "c↓", r, "c↑", s)
        end
      end
    end
  end
  return ham
end

function gen_occupation_state(occa, occb, norb)
  state = ones(Int, norb)
  for oa in occa
    state[oa] += 1
  end
  for ob in occb
    state[ob] += 2
  end
  return state
end

"""
    calc_dmrg(EC::ECInfo)

  Perform DMRG calculation
"""
function calc_dmrg(EC::ECInfo)
  print_info("DMRG")
  SP = EC.space
  hamiltonian = gen_hamiltonian(EC)
  ref_state = gen_occupation_state(SP['o'], SP['O'], n_orbs(EC))
  println("Number of orbitals in DMRG: ", length(ref_state))

  sites = siteinds("Electron", length(ref_state); conserve_qns=true)

  println("\nConstruct MPO")

  H = @time MPO(hamiltonian, sites)
  println("MPO constructed")

  println("Maximum link dimension: ", maxlinkdim(H))

  ψref = MPS(sites, ref_state)
  Eref = inner(ψref', H, ψref)
  println("Reference energy: ", Eref)

  dmrg_params = (nsweeps=EC.options.dmrg.nsweeps, maxdim=EC.options.dmrg.maxdim, 
                cutoff=EC.options.dmrg.cutoff, noise=EC.options.dmrg.noise)

  println("\nRunning DMRG")
  @show dmrg_params
  E, ψ = dmrg(H, ψref; dmrg_params...)
  println("DMRG complete")
  E2 = inner(ψ', H, ψ)
  return OutDict("E"=>E-Eref, "Expect"=>E2-Eref)
end

end # module DMRG
