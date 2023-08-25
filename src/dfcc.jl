"""
  DFCoupledCluster

  Density-fitted coupled-cluster methods.
"""
module DFCoupledCluster
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.DIIS

export calc_dressed_3idx, calc_svd_dcsd_residual


"""
    get_endauxblks(naux, blocksize = 100)

  Generate end-of-block indices for auxiliary basis (for loop over blocks).
"""
function get_endauxblks(naux, blocksize = 100)
  nauxblks = naux ÷ blocksize
  if nauxblks == 0 || naux - nauxblks*blocksize > 0.5*blocksize
    nauxblks += 1
  end
  endauxblks = [ (i == nauxblks) ? naux : i*blocksize for i in 1:nauxblks ]
  return endauxblks
end

"""
    calc_dressed_3idx(EC,T1)

  Calculate dressed integrals for 3-index integrals from file `pqP`.
"""
function calc_dressed_3idx(EC,T1)
  pqPfile, pqP = mmap(EC, "pqP")
  # println(size(pqP))
  SP = EC.space
  nP = size(pqP,3)
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  # create mmaps for dressed integrals
  ovPfile, ovP = newmmap(EC,"d_ovP",Float64,(nocc,nvirt,nP))
  voPfile, voP = newmmap(EC,"d_voP",Float64,(nvirt,nocc,nP))
  ooPfile, ooP = newmmap(EC,"d_ooP",Float64,(nocc,nocc,nP))
  vvPfile, vvP = newmmap(EC,"d_vvP",Float64,(nvirt,nvirt,nP))

  PBlks = get_endauxblks(nP)
  sP = 1 # start index of each block
  for eP in PBlks # end index of each block
    P = sP:eP
    ovP[:,:,P] = pqP[SP['o'],SP['v'],P]
    vvP[:,:,P] = pqP[SP['v'],SP['v'],P]
    @tensoropt vvP[:,:,P][a,b,P] -= T1[a,i] * ovP[:,:,P][i,b,P]
    voP[:,:,P] = pqP[SP['v'],SP['o'],P]
    @tensoropt voP[:,:,P][a,i,P] += T1[b,i] * vvP[:,:,P][a,b,P]
    ooP[:,:,P] = pqP[SP['o'],SP['o'],P]
    @tensoropt voP[:,:,P][a,i,P] -= T1[a,j] * ooP[:,:,P][j,i,P]
    @tensoropt ooP[:,:,P][i,j,P] += T1[b,j] * ovP[:,:,P][i,b,P]
    sP = eP + 1
  end
  closemmap(EC,ovPfile,ovP)
  closemmap(EC,voPfile,voP)
  closemmap(EC,ooPfile,ooP)
  closemmap(EC,vvPfile,vvP)
  close(pqPfile)
end

"""
    calc_svd_dcsd_residual(EC::ECInfo, T1, T2_XY)

  Calculate decomposed closed-shell DCSD residual with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` and
  ``R_{XY}=U^{iX†}_a U^{jY†}_b R^{ij}_{ab}``.
"""
function calc_svd_dcsd_residual(EC::ECInfo, T1, T2_XY)
  t1 = time_ns()
  SP = EC.space
  if length(T1) > 0
    # calc_dressed_ints(EC,T1)
    t1 = print_time(EC,t1,"dressing",2)
  else
  end
end

end # module DFCoupledCluster