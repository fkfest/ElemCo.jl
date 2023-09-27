# example: calculate DCSD/BO-HF/DCSD for all FCIDUMPs in "./*/" folders
# create a symbolic link in the current directory pointing the ElemCo.jl-devel folder
# ln -s <path-to-ElemCo.jl-devel-folder> .

include("ElemCo.jl-devel/src/ElemCo.jl")
using .ElemCo
using .ElemCo.ECInfos
using .ElemCo.BOHF
using .ElemCo.FciDump

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
  output = "bo-dcsd.out"
  redirect_stdio(stdout=output) do
    @cc dcsd fcidump="FCIDUMP"

    # to do directly BO-HF without calculating dcsd uncomment next lines
    # EC=ECInfo()
    # EC.fd = read_fcidump(fcidump)
    if ElemCo.is_closed_shell(EC)[1]
      EBOHF = bohf(EC)
    else
      EBOHF = bouhf(EC)
    end
    CMOr = load(EC, EC.options.wf.orb)
    CMOl = load(EC, EC.options.wf.orb*EC.options.wf.left)
    transform_fcidump(EC.fd, CMOl, CMOr)
    EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump="")
  end
  cd("..")
end


