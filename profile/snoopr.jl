# start julia as `jlm --startup-file="no" `
using SnoopCompileCore
using ElemCo

invalidations = @snoopr begin

geometry="H 0.0 0.0 0.0
          H 0.0 0.0 1.0"
basis="vdz"
@dfhf
@cc dcsd
end

using SnoopCompile

