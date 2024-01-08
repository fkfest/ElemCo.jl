# example: calculate DCSD/BO-HF/DCSD for all FCIDUMPs in "./*/" folders
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
    output = "bo-dcsd.out"
    redirect_stdio(stdout=output) do
      fcidump="FCIDUMP"
      @cc dcsd
      EBOHF = @bohf
      @transform_ints biorth
      @cc dcsd
    end
  end
  cd("..")
end


