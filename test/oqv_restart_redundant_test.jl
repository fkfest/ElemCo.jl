@testitem "oqv_restart_redundant" tags=[:qvcc, :quick] begin
using ElemCo
using Test

# Restart OQV-DCD into a REDUNDANT (linearly-dependent) basis via dump=""+start. The redundant
# orbitals of the new basis must be handled exactly like a fresh (DF-)HF: deleted and excluded from
# the correlation treatment. Since the extra (duplicated) functions add nothing, the restarted energy
# must recover the non-redundant one.
geometry = "bohr
     H1 0.0 0.0 0.0
     H2 0.0 0.0 1.4"
hbasis_nonredund = "{
  s, H, 13.01, 1.962, 0.4446, 0.122
  c, 1.4, 0.019685, 0.137977, 0.478148, 0.50124
  c, 4.4, 1.0
  p, H, 0.727
  c, 1.1, 1.0}"
# one duplicated s contraction -> exactly 2 redundant functions for H2
hbasis_redund = "{
  s, H, 13.01, 1.962, 0.4446, 0.122
  c, 1.4, 0.019685, 0.137977, 0.478148, 0.50124
  c, 4.4, 1.0
  c, 4.4, 1.0
  p, H, 0.727
  c, 1.1, 1.0}"

tmp = mktempdir()
wf_n = joinpath(tmp, "wf_n.h5"); cc_n = joinpath(tmp, "cc_n.h5")

@set scf redthr=1.e-6
basis = Dict("ao"=>hbasis_nonredund, "jkfit"=>"cc-pvdz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
@set wf dump=wf_n
@dfhf
E_store = (@cc oqv-dcd begin @set wf dump=wf_n store=cc_n end)["OQV-DCD"]

basis = Dict("ao"=>hbasis_redund, "jkfit"=>"cc-pvdz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
E_restart = (@cc oqv-dcd begin @set wf start=cc_n dump="" end)["OQV-DCD"]

@testset "OQV-DCD restart into a redundant basis (dump=\"\")" begin
  @test abs(E_restart - E_store) < 1.e-6
end
rm(tmp; force=true, recursive=true)
end
