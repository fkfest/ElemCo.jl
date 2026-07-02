@testitem "oqv_restart" tags=[:qvcc, :quick] begin
using ElemCo
using Test

# Restart an orbital-optimized CC (OQV-DCD) reusing the stored optimized orbitals.
# `dump=""` tells the restart there is no separate reference dump, so the FCIDUMP is
# rebuilt from the optimized orbitals in `start` (cc.h5) and the amplitudes are read from
# there directly, instead of projecting them onto the (HF) orbitals of a separate dump.
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"sto-3g", "jkfit"=>"cc-pvdz-jkfit", "mpfit"=>"cc-pvdz-mpfit")

@testset "OQV-DCD restart reusing optimized orbitals (dump=\"\")" begin
  tmpdir = mktempdir()
  ccfile = joinpath(tmpdir, "cc.h5")

  @dfhf
  e1 = @cc oqv-dcd begin
    @set wf store=ccfile
  end
  @test isfile(ccfile)

  # Reuse the optimized orbitals: no separate reference dump, read orbitals+amplitudes
  # from cc.h5. Without the fix this crashed in dfdump (TREXIO_OPEN_ERROR on empty dump).
  e2 = @cc oqv-dcd begin
    @set wf start=ccfile dump=""
  end

  @test abs(e1["OQV-DCD"] - e2["OQV-DCD"]) < 1.e-6
  rm(tmpdir; force=true, recursive=true)
end
end
