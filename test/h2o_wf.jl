using ElemCo
using LinearAlgebra

@testset "H2O WF Test" begin
epsilon    =   1.e-6

geometry = "
     O      0.000000000    0.000000000   -0.130186067
     H      0.000000000    1.489124508    1.033245507
     H      0.000000000   -1.489124508    1.033245507
     O1     4.000000000    0.000000000   -0.130186067
     H1     4.000000000    1.489124508    1.033245507
     H1     4.000000000   -1.489124508    1.033245507"

basis = "vdz"

@print_input

@dfhf

orbs = @loadwf

SAO = ElemCo.Integrals.overlap(orbs["basis"])
CMO = orbs["orbitals"][1]
@test norm(CMO'*SAO*CMO - I) < epsilon

wf_dir = mktempdir()  
wf_path = joinpath(wf_dir, "mywf.h5")  
try
  @dummy ["O1", "H1", "H1"]
  ehf = @dfhf
  @copywf wf_path
  @dfints begin
    @set wf freeze_nvirt=10
  end
  ehf2 = @bohf
  @test abs(last_energy(ehf)-last_energy(ehf2)) < epsilon

  en1 = @cc dcsd
  @usewf wf_path
  @dfints begin
    @set wf freeze_nvirt=10
  end
  en2 = @cc dcsd
  @test abs(last_energy(en1)-last_energy(en2)) < epsilon
finally  
  if ispath(wf_path)  
    rm(wf_path; force=true)  
  end  
end 

end
