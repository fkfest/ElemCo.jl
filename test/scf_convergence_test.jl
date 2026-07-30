@testitem "scf_convergence" tags=[:scf, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.MSystems: parse_geometry
using ElemCo.TensorTools: load2idx
using LinearAlgebra

geometry = "bohr
    O      0.000000000    0.000000000   -0.130186067
    H1     0.000000000    1.489124508    1.033245507
    H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-rifit")

# each EC gets its own orbital dump in its (unique) scratch dir, so the runs below don't
# clash over the shared default "wf.h5" in the working directory
fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, basis));
           e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)
orb_energies(ec) = ElemCo.OrbTools.fetch_orbital_energies(ec, "mo")[1]

# max |off-diagonal| of the occ-occ and virt-virt Fock blocks the correlated methods see —
# exactly what the (T) diagonality check (`cc.fock_diag_thr`) looks at
function max_fock_offdiag(ec, closed_shell)
  ec.options.wf.freeze_nocc = 0
  orbs = ElemCo.OrbTools.load_orbitals_for_correlation(ec)
  ElemCo.CoupledCluster.ao_cc_setup!(ec; closed_shell=closed_shell, orbitals=orbs)
  off(f, sp) = isempty(sp) ? 0.0 : maximum(abs.(f[sp,sp] - Diagonal(diag(f[sp,sp]))); init=0.0)
  res = 0.0
  for (key, o, v) in (("f_mm", 'o', 'v'), ("f_MM", 'O', 'V'))
    file_exists(ec, key) || continue
    f = load2idx(ec, key)
    res = max(res, off(f, ec.space[o]), off(f, ec.space[v]))
  end
  return res
end

@testset "scf.maxit=0 builds the Fock matrix without iterating" begin
  # `maxit=0` used to leave the Fock matrix the SCF loops return undefined
  # (`UndefVarError: fock`), because the iteration body never ran. It must instead build the
  # Fock matrix (and the energy) once for the given orbitals and leave those orbitals alone.
  # The energy is then that of the (unconverged) guess, well above the converged HF, and the
  # stored orbital energies are the Fock expectation values ⟨p|F|p⟩ rather than zeros.
  EC = fresh()
  e_hf = @hf begin
    @set scf maxit=0
  end
  @test isfinite(last_energy(e_hf)) && -76.0 < last_energy(e_hf) < 0.0
  @test !all(iszero, orb_energies(EC))
  # the guess orbitals were left untouched, so they do NOT diagonalize the Fock matrix that was
  # built from them (the canonicalization below applies to converged runs only)
  @test max_fock_offdiag(EC, true) > 1.e-3

  EC = fresh()
  e_uhf = @uhf begin
    @set scf maxit=0
  end
  @test isfinite(last_energy(e_uhf)) && -76.0 < last_energy(e_uhf) < 0.0
  @test !all(iszero, orb_energies(EC))

  EC = fresh()
  e_dfhf = @dfhf begin
    @set scf maxit=0
  end
  @test isfinite(last_energy(e_dfhf)) && -76.0 < last_energy(e_dfhf) < 0.0
  @test !all(iszero, orb_energies(EC))

  EC = fresh()
  e_dfuhf = @dfuhf begin
    @set scf maxit=0
  end
  @test isfinite(last_energy(e_dfuhf)) && -76.0 < last_energy(e_dfuhf) < 0.0
  @test !all(iszero, orb_energies(EC))

  # a no-iteration run fed the converged orbitals reproduces the converged solution exactly:
  # it is precisely one Fock build for those orbitals
  EC = fresh()
  econv = @hf
  eps_conv = copy(orb_energies(EC))
  e0 = @hf begin
    @set scf maxit=0
    @set wf start=EC.options.wf.dump
  end
  @test abs(last_energy(e0) - last_energy(econv)) < 1.e-12
  @test maximum(abs.(orb_energies(EC) .- eps_conv)) < 1.e-8
  # the orbitals it read back are the canonicalized ones of the converged run, so the Fock
  # matrix built from them is diagonal — the canonicality survives the orbital dump round trip
  @test max_fock_offdiag(EC, true) < 1.e-11
end

@testset "converged orbitals are canonical (no pseudo-canonicalization needed)" begin
  # The SCF loop builds the Fock matrix, tests convergence and breaks, so its orbitals
  # diagonalize the (DIIS-extrapolated) Fock matrix of the PREVIOUS iteration. The final Fock
  # matrix then kept occ-occ/virt-virt off-diagonal elements of the order of the remaining
  # orbital gradient (≈0.1·sqrt(scf.thr), i.e. ~1e-6 at the default scf.thr=1e-10) — enough to
  # trip the `cc.fock_diag_thr` check and force a pseudo-canonicalization in (T). The loops now
  # canonicalize within the occupied and within the virtual space, which leaves the density
  # (hence the energy and the Fock matrix itself) exactly invariant.
  EC = fresh()
  @hf
  @test max_fock_offdiag(EC, true) < 1.e-11

  # also for a deliberately loosely converged SCF: the canonicality no longer tracks scf.thr
  EC = fresh()
  @hf begin
    @set scf thr=1.e-8
  end
  @test max_fock_offdiag(EC, true) < 1.e-11

  # open shell (water cation): both spin blocks
  EC = fresh()
  @set wf charge=1 ms2=1
  @uhf
  @test max_fock_offdiag(EC, false) < 1.e-11
end

@testset "(T) needs no pseudo-canonicalization on converged HF orbitals" begin
  # With the diagonality check disabled (`fock_diag_thr < 0` skips it and takes the orbitals to
  # be canonical) the triples energy must be unchanged — it would be wrong if they were not.
  EC = fresh()
  @hf
  e_checked = @cc "ccsd(t)"
  EC = fresh()
  @hf
  e_skipped = @cc "ccsd(t)" begin
    @set cc fock_diag_thr=-1.0
  end
  @test abs(e_skipped["CCSD(T)"] - e_checked["CCSD(T)"]) < 1.e-10
end
end
