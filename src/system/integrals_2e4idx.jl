# 2-electron 4-index integrals
# adapted from GaussianBasis.jl

function eri_2e4idx!(out, i, j, k, l, bs::BasisSet)
  cint2e_sph!(out, [i,j,k,l], bs.lib)
end
