using ElemCo

function main()
  geometry="angstrom
                              O     1.2091536548    1.7664118189   -0.0171613972
                              H     2.1984800075    1.7977100627    0.0121161719
                              H     0.9197881882    2.4580185570    0.6297938832"
  basis="vdz"
  @time @dfhf
  @time @cc dcsd
end
@time main()
