@testitem "oqv_restart_dump4core" tags=[:qvcc, :quick] begin
using ElemCo
using Test

# Issue #357: on a geometry-change restart of an orbital-optimized method, the frozen-core orbitals
# reused from `start` are stale — stuck at the *previous* geometry — which corrupts the reference
# (for stretched N2 the error is ~0.07 Ha and cascades along a curve). `dump4core_only=true` fixes
# this: the frozen core is taken from `dump` (a fresh HF at the *current* geometry) while the
# correlating orbitals + amplitudes are still reused from `start` (projected, and orthogonalized
# against the new core by Gram–Schmidt). The restart should then reproduce a from-scratch calculation.
basis = "vdz"

@testset "OQV-DCD dump4core_only restart across a geometry change" begin
  tmp = mktempdir()
  start = joinpath(tmp, "cc.h5")   # OQV-DCD solution at the previous geometry R1 = 1.5 Å
  hf = joinpath(tmp, "hf.h5")      # fresh HF at the new geometry R2 = 1.6 Å (the `dump`)

  # previous geometry: build the OQV "start" to restart from
  geometry = "angstrom
       N 0.0 0.0 0.0
       N 1.5 0.0 0.0"
  @dfhf
  @cc oqv-dcd begin
    @set wf store=start
    @set cc thr=1.0e-8
  end

  # new geometry: from-scratch reference
  geometry = "angstrom
       N 0.0 0.0 0.0
       N 1.6 0.0 0.0"
  @dfhf
  eref = @cc oqv-dcd begin
    @set cc thr=1.0e-8
  end

  # new geometry restart: reuse R1's optimized orbitals + amplitudes (projected), but take the frozen
  # core from a fresh R2 HF (`dump`). Without `dump4core_only` this restart lands ~1.2 Ha off (stale core).
  @dfhf begin
    @set wf store=hf
  end
  ecd = @cc oqv-dcd begin
    @set wf dump=hf start=start dump4core_only=true
    @set cc thr=1.0e-8
  end

  @test abs(ecd["OQV-DCD"] - eref["OQV-DCD"]) < 1.0e-3
  rm(tmp; force=true, recursive=true)
end
end
