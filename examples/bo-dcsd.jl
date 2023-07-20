# example: calculate DCSD/BO-HF/DCSD for all FCIDUMPs in "./*/" folders
# modify include path to point to the ElemCo.jl-devel folder

include("ElemCo.jl-devel/src/ElemCo.jl")
using .ElemCo
using .ElemCo.ECInfos
using .ElemCo.BOHF
using .ElemCo.FciDump

skip = false
# if uncommented: skip all folders upto some specific folder 
#skip = true
for dir in readdir()
  if skip
    if dir == "H2O2"
      global skip = false
    else
      continue
    end
  end
  if !isdir(dir) || dir == "H" #atm skip H
    continue
  end
  println(dir)
  cd(dir)
  fcidump = "FCIDUMP"
  output = "bo-dcsd.out"
  EC = ECInfo()
  redirect_stdio(stdout=output) do
    EHF, EMP2, ECCSD = ECdriver(EC, "dcsd"; fcidump)

    # to do directly BO-HF without calculating dcsd uncomment next line
    #setup_scratch_and_fcidump(EC, fcidump)
    if ElemCo.is_closed_shell(EC)[1]
      EBOHF, ϵ,CMOl,CMOr = bohf(EC)
    else
      EBOHF, ϵ,CMOl,CMOr = bouhf(EC)
    end
    display(ϵ)
    transform_fcidump(EC.fd, CMOl, CMOr)
    EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump="")
  end
  cd("..")
end


