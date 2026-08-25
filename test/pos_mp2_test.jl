# NOTE: pre-existing failure ("Cannot have positrons and spin > 0").
# This file was not referenced by the old runtests.jl and never ran.
# Tagged :broken so it is skipped by CLI runs but stays visible in the
# VS Code Test Explorer. Remove the :broken tag once fixed.
@testitem "pos_mp2" tags=[:pos, :broken] begin
using ElemCo
@print_input

geometry="bohr
            Li 0.000000 0.000000 0.000000
            H  0.000000 0.000000 3.0196
            H2 0.000000 0.000000 6.0196"


basis = Dict("ao"=>"aug-cc-pvdz",
     "jkfit"=>"def2-universal-jkfit",
     "mpfit"=>"aug-cc-pv5z-rifit")

@ECinit
@set wf charge=0
@set wf npositron=1
@set wf freeze_nocc=0
E_LiH=@dfhf
@dfints
@write_ints
@cc mp2



end
