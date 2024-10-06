# example: calculate DCSD for all FCIDUMPs in "./*/" folders
using ElemCo
@print_input
geometry="bohr
     Be 0.000000 0.000000 0.000000
     O  0.000000 0.000000 2.515"


basis="def2-sv(p)"

@dfhf

