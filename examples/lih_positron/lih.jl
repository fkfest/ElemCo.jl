# example: calculate DCSD for all FCIDUMPs in "./*/" folders
using ElemCo
@print_input
geometry="bohr
     Li 0.000000 0.000000 0.000000
     H  0.000000 0.000000 3.0196"


#basis="def2-sv(p)"
#basis="vdz"
basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"def2-universal-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")
@set scf guess=:HCORE

@dfhf_positron

