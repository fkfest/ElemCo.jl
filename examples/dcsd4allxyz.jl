# example: calculate DCSD for all xyz files in the current directory
# create a symbolic link in the current directory pointing the ElemCo.jl-devel folder
# ln -s <path-to-ElemCo.jl-devel-folder> .

include("ElemCo.jl-devel/src/ElemCo.jl")
using .ElemCo

# if non-empty list: calculate only specified xyz files
calc_only = []
# don't calculate the following xyz files
dont_calc = ["Li2.xyz"]
for file in readdir()
  if length(calc_only) > 0 && file ∉ calc_only
    continue
  end
  if !isfile(file) || file in dont_calc
    continue
  end
  mainname, extension = @mainname(file) 
  if extension != "xyz"
    continue
  end
  println(mainname)
  output = mainname*".out"
  redirect_stdio(stdout=output) do
    geometry = file
    basis = Dict("ao"=>"cc-pVDZ",
            "jkfit"=>"cc-pvtz-jkfit",
            "mp2fit"=>"cc-pvtz-rifit")
    @dfhf
    @cc dcsd
  end
end


