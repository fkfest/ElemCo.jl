# start julia as `jlm --startup-file="no" `
using SnoopCompile
using Profile
using ElemCo


geometry="H 0.0 0.0 0.0
          H 0.0 0.0 1.0"
basis="vdz"
tinf = @snoopi_deep begin
  @dfhf
  @cc dcsd
end

@profile begin
  @dfhf
  @cc dcsd
end

