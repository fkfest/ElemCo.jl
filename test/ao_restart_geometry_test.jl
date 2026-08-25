@testitem "ao_restart_geometry" tags=[:cc, :pm, :quick] begin
using ElemCo
using Test

# Reusing orbitals across a geometry change on the EXACT-AO routes. `wf.dump=""` + `wf.start` makes
# `dumpfile` fall back to the start file, so the correlated calculation silently picks up orbitals
# from the previous geometry — they must be projected onto the current AO basis AND re-orthonormalized
# in its metric, i.e. read through `load_orbitals_for_correlation` (as the DF/FCIDUMP route does),
# not the bare `load_orbitals`.
#
# Regression: both exact-AO routes used `load_orbitals`, which projects but does not
# re-orthonormalize. The AO-direct route then ran on a reference with max|CᵀSC − I| ≈ 2e-5 and
# silently returned a CCSD energy 6.9e-4 Eh off; the derive route (int.ao_direct=false) hit
# `generate_mo_dump`'s orthonormality assertion and died. Both are checked here: they must agree with
# each other, and — since a re-orthonormalized projection is a well-defined reference — be stable.
G1 = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
# displaced by 1e-3 bohr: enough that the projected orbitals are no longer orthonormal in the new
# metric, small enough that the energies stay comparable
G2 = "bohr
     O      0.001000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = "vdz"

@testset "exact-AO restart across a geometry change (dump=\"\"+start)" begin
  tmp = mktempdir()
  ccfile = joinpath(tmp, "cc.h5")

  # store a wavefunction at G1 (AO-direct route)
  geometry = G1
  @ints
  @hf
  @cc ccsd begin @set wf store=ccfile end

  # restart at the displaced geometry G2, reusing the G1 orbitals
  geometry = G2
  @ints
  @hf
  e_ao = @cc ccsd begin @set wf start=ccfile dump="" end

  # the same restart through the derived-MO-dump route, which asserts orthonormality in
  # `generate_mo_dump` — before the fix this threw instead of producing a number
  e_mo = @cc ccsd begin
    @set wf start=ccfile dump=""
    @set int ao_direct=false
  end
  @test abs(e_ao["CCSD"] - e_mo["CCSD"]) < 1e-9
  @test abs(e_ao["HF"] - e_mo["HF"]) < 1e-9

  rm(tmp; force=true, recursive=true)
end

# `wf.dump4core_only` on the exact-AO routes: the correlating orbitals come from `start` (optimized
# at the previous geometry) while the frozen core is taken from `dump` (a fresh HF at the current
# one) and the correlating orbitals are re-orthonormalized against it. Previously the AO routes
# silently ignored the option and simply used the `dump` orbitals throughout.
#
# What this asserts is that the restart CONVERGES to the from-scratch answer (as the DF-route test
# oqv_restart_dump4core does). It deliberately does not claim to fail without the implementation:
# with a non-empty `dump` the ignored path also starts from valid current-geometry orbitals and
# reaches the same minimum — the point of the option is that the optimized orbitals/amplitudes are
# genuinely reused rather than discarded, not a different converged energy.
@testset "AO-direct dump4core_only restart reaches the from-scratch solution" begin
  tmp = mktempdir()
  start = joinpath(tmp, "cc.h5")   # OQV-DCD solution at the previous geometry
  hffile = joinpath(tmp, "hf.h5")  # fresh HF at the new geometry (the `dump`)

  geometry = "angstrom
       N 0.0 0.0 0.0
       N 1.5 0.0 0.0"
  @ints
  @hf
  @cc oqv-dcd begin
    @set wf store=start
    @set cc thr=1.0e-8
  end

  geometry = "angstrom
       N 0.0 0.0 0.0
       N 1.6 0.0 0.0"
  @ints
  @hf
  eref = @cc oqv-dcd begin @set cc thr=1.0e-8 end

  @ints
  @hf begin @set wf store=hffile end
  ecd = @cc oqv-dcd begin
    @set wf dump=hffile start=start dump4core_only=true
    @set cc thr=1.0e-8
  end
  @test abs(ecd["OQV-DCD"] - eref["OQV-DCD"]) < 1.0e-3

  rm(tmp; force=true, recursive=true)
end
end
