
"""
    test_dressed_ints(EC,T1)

  Compare 3-idx dressed integrals to 4-idx dressed integrals.
"""
function test_dressed_ints(EC,T1)
  calc_dressed_ints(EC,T1)
  ooPfile, ooP = mmap(EC,"d_ooP")
  vvPfile, vvP = mmap(EC,"d_vvP")
  @tensoropt abij[a,b,i,j] := vvP[a,b,P] * ooP[i,j,P]
  close(vvPfile)
  if isapprox(permutedims(abij,(1,3,2,4)), load(EC,"d_vovo"), atol = 1e-6)
    println("dressed integrals (ab|ij) ok")
  else
    println("dressed integrals (ab|ij) not ok")
  end
  voPfile, voP = mmap(EC,"d_voP")
  @tensoropt aijk[a,i,j,k] := voP[a,i,P] * ooP[j,k,P]
  if isapprox(permutedims(aijk,(1,3,2,4)), load(EC,"d_vooo"), atol = 1e-4)
    println("dressed integrals (ai|jk) ok")
  else
    println("dressed integrals (ai|jk) not ok")
  end
  ovPfile, ovP = mmap(EC,"d_ovP")
  @tensoropt aijb[a,i,j,b] := voP[a,i,P] * ovP[j,b,P]
  if isapprox(permutedims(aijb,(1,3,2,4)), load(EC,"d_voov"), atol = 1e-6)
    println("dressed integrals (ai|jb) ok")
  else
    println("dressed integrals (ai|jb) not ok")
  end
  close(ovPfile)
  close(voPfile)
  close(ooPfile)
end

"""
    test_add_to_singles_and_doubles_residuals(R1,R2,T1,T2)

  Test R1(T3) and R2(T3)
"""
function test_add_to_singles_and_doubles_residuals(R1,R2,T1,T2) 
  @tensoropt ETb3 = (2.0*T2[a,b,i,j] - T2[b,a,i,j]) * R2[a,b,i,j]
  println("ETb3: ",ETb3)
  @tensoropt ETT1 = 2.0*T1[a,i] * R1[a,i]
  println("ETT1: ",ETT1)
end

"""
    test_calc_pertT_from_T3(EC,T3)

  Test [T]
"""
function test_calc_pertT_from_T3(EC::ECInfo, T3)
  nocc = length(EC.space['o'])
  nvirt = length(EC.space['v'])
  ϵo, ϵv = orbital_energies(EC)
  # test [T]
  Enb3 = 0.0
  for i = 1:nocc
    for j = 1:nocc
      for k = 1:nocc
        for a = 1:nvirt
          for b = 1:nvirt
            for c = 1:nvirt
              W = (T3[a,i,b,j,c,k] * (ϵv[a] + ϵv[b] + ϵv[c] - ϵo[i] - ϵo[j] - ϵo[k]))
              Enb3 += W*(4/3*T3[a,i,b,j,c,k]-2.0* T3[a,i,b,k,c,j]+2/3*T3[c,i,a,j,b,k])
            end
          end
        end
      end
    end
  end
  println("Enb3: ",Enb3)
end

"""
    test_UaiX(EC,UaiX)

  Test UaiX
"""
function test_UaiX(EC::ECInfo, UaiX)
  nocc = length(EC.space['o'])
  nvirt = length(EC.space['v'])
  rescaledU = deepcopy(UaiX)
  ϵo, ϵv = orbital_energies(EC)
  for a in 1:nvirt
    for i in 1:nocc
      rescaledU[a,i,:] *= (ϵv[a] - ϵo[i])
    end
  end

  @tensoropt begin
    TestIntermediate1[X,Y] := UaiX[a,i,X] * rescaledU[a,i,Y]
  end
  if TestIntermediate1 ≈ diagm(load(EC,"epsilonX"))
    println("UaiX ok")
  else
    println("UaiX not ok")
  end
end
