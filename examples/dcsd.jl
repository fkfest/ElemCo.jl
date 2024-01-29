# example: calculate DCSD for all FCIDUMPs in "./*/" folders
using ElemCo

# if non-empty list: calculate only specified folders
calc_only = []
# don't calculate the following folders
dont_calc = []
for dir in readdir()
  if length(calc_only) > 0 && dir ∉ calc_only
    continue
  end
  if !isdir(dir) || dir in dont_calc
    continue
  end
  println(dir)
  cd(dir)
  if isfile("FCIDUMP")
    output = "dcsd.out"
    redirect_stdio(stdout=output) do
      @cc dcsd fcidump="FCIDUMP"
    end
  end
  cd("..")
end


