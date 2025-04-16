using ElemCo

geometry="bohr
            Li 0.000000 0.000000 0.000000
            H  0.000000 0.000000 3.0196"


basis = Dict("ao"=>"sto-3g",
     "jkfit"=>"sto-3g",
     "mp2fit"=>"sto-3g","mpfit"=>"sto-3g")

@ECinit
@set wf charge=0
@set wf npositron=1
E_LiH=@dfhf
@dfints


