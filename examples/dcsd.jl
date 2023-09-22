# example: calculate DCSD for all FCIDUMPs in "./*/" folders
# create a symbolic link in the current directory pointing the ElemCo.jl-devel folder
# ln -s <path-to-ElemCo.jl-devel-folder> .

include("ElemCo.jl-devel/src/ElemCo.jl")
using .ElemCo

# if non-empty list: calculate only specified folders
calc_only = []
# don't calculate the following folders
dont_calc = ["ElemCo.jl-devel"]
for dir in readdir()
  if length(calc_only) > 0 && dir ∉ calc_only
    continue
  end
  if !isdir(dir) || dir in dont_calc
    continue
  end
  println(dir)
  cd(dir)
  output = "dcsd.out"
  redirect_stdio(stdout=output) do
    @cc dcsd fcidump="FCIDUMP"
  end
  cd("..")
end


