"""
This file contains all relevant functions of EOM-SVD-DCSD
"""

"""
    calc_svd_eom(EC::ECInfo, method::ECMethod)

  Calculate decomposed closed-shell EOM-DCSD.

  Currently only DC methods are supported.
  DF integrals are used (have to be calculated before).
"""
function calc_svd_eom(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  nstates = EC.options.eom.nstates
  shift = 0
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  ϵo, ϵv = orbital_energies(EC) 
  #shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=false)
  states = [1:nstates;]
  
  energies = calc_df_eom(EC,ECMethod("EOM-CCS"))
  println("*******************************************************************************************")
  display(energies)
  #calc_eom(EC,ECMethod("EOM-CCS"))  

  println("Hello")

    
  U1 = zeros(nvirt, nocc)
  U2 = zeros(nvirt, nvirt, nocc, nocc)
  Vecs = (U1, U2)
# load the CIS eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level, false)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      if file_exists(EC, filename)
        println("Reading $descr from file $filename")
        load!(EC, filename, U1)
        add_trial_vector!(dav, Vecs, st)
        #display(U1)
        
        R2 = calc_R2_df_cis_pert_d(EC, U1)
        #display(R2)        

        #U2 = zeros(nvirt,nvirt,nocc,nocc)
        #display(U2)
        new_doubles_trial!(R2, ϵo, ϵv, energies[st], 0) #replaces R2 with U2       
        #display(U2)        
        
        #excitation_level_2 = 2
        #mainfilename, descr = save_or_start_file(EC, "X", excitation_level_2)
        #if mainfilename != "" 
        #  filename = mainfilename*"_$excitation_level_2"*"^$st"
        #  println("Saving $descr for state $st to $filename")
        #  
        #  save!(EC, filename, U2, description=descr) 
        #end       
 
        UaiX = svd_decompose(reshape(permutedims(R2, (1,3,2,4)), (nvirt*nocc,nvirt*nocc)), nvirt, nocc, sqrt(EC.options.cc.ampsvdtol)) 
        #display(UaiX) 
        save!(EC, "C_voX^$st", UaiX)        

      else
        error("File $filename not found, cannot read CIS eigenvector")
      end
    end
  else
    error("No file found for CIS eigenvectors")
  end

  
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, Vecs, st)
      V1, V2 = calc_eom_svd_au(EC, Vecs..., st) 
      add_product_vector!(dav, (V1,V2), st)
    end
    energies = perform!(dav)
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, Vecs, st)
      NormR1 = calc_singles_norm(Vecs[1])
      NormR2 = calc_doubles_norm(Vecs[2])
      NormR = NormR1 + NormR2
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        new_trial_vector!(EC, Vecs, energies[st])
        add_trial_vector!(dav, Vecs, st)
        push!(states2do, st)
      end
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end



## store the final eigenvectors
#  excitation_level = 1
#  mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
#  if mainfilename != ""
#    for st in 1:nstates
#      filename = mainfilename*"_$excitation_level"*"^$st"
#      println("Saving $descr for state $st to $filename")
#      U11 = rand([0, 1], nvirt, nocc)
#      display(U11)
#      save!(EC, filename, U11, description=descr)
#    end
#  end


#  U11 = zeros(nvirt, nocc)
## load the CIS eigenvectors
#  excitation_level = 1
#  mainfilename, descr = save_or_start_file(EC, "X", excitation_level, false)
#  if mainfilename != ""
#    for st in 1:nstates
#      filename = mainfilename*"_$excitation_level"*"^$st"
#      if file_exists(EC, filename)
#        println("Reading $descr from file $filename")
#        load!(EC, filename, U11)
#        display(U1)
#      else
#        error("File $filename not found, cannot read CIS eigenvector")
#      end
#    end
#  else
#    error("No file found for CIS eigenvectors")
#  end



  #Achtung, hier wird nur einer der U Vektoren von vielen gespeichert...
  #U1_1 = load2idx(EC, "1U1")
  #display(U1_1)
  #U1_2 = load2idx(EC, "2U1")
  #display(U1_2)

  #R2 = calc_R2_df_cis_pert_d(EC, U1_1)
  #display(R2)
  #println("Hello3")

  #calc_eom_svd_au(EC)
end


"""
    calc_eom_svd_au(EC::ECInfo)

  Calculate singles and doubles AUs for SVD-EOM-DCSD.
"""
function calc_eom_svd_au(EC::ECInfo,U1, U2, st)
  println("Hello2")
  t1 = time_ns()
  mem1 = free_memory()
  U_voX = load3idx(EC, "C_voX")
  @mtensor U_oXv[k,X,c] := U_voX[c,k,X]
  bU_vobX = load3idx(EC, "C_voX^$st")
  @mtensor bU_obXv[k,bX,c] := bU_vobX[c,k,bX]
  #display(UvoX)
  
  T2 = load4idx(EC,"T_vvoo")
  @mtensor U_bXbX[bX,bY] := U2[a,b,i,j] * bU_obXv[i,bX,a] * bU_obXv[j,bY,b]   
 
  #only for testing
  #@buffer buf(lenbuf) begin
  #@mtensor U_oXv[k,X,c] := U_voX[c,k,X]
  #@mtensor bU_obXv[k,X,c] := U_voX[c,k,X]
  #@mtensor bU_vobX[c,k,X] := U_voX[c,k,X]
  #end
  
  #load decomposed amplitudes
  T_XX = load2idx(EC, "T_XX")
  #display(T_XX)
  
  #only for testing
  #U_bXbX = load2idx(EC, "U_{bX}{bX}")
  #@mtensor U_bXbX[X,Y] := T_XX[X,Y]  
  #T2 = load4idx(EC,"T_vvoo")
  #@mtensor T2[a,b,i,j] := T_XX[X,Y] * U_voX[a,i,X] * U_voX[b,j,Y] 
  #@mtensor U2[a,b,i,j] := T_XX[X,Y] * U_voX[a,i,X] * U_voX[b,j,Y]   

  #load df coeff
  ovLfile, v_ovL = mmap3idx(EC, "d_ovL") #should be same as non-dressed ovL
  voLfile, dv_voL = mmap3idx(EC, "d_voL")
  ooLfile, dv_ooL = mmap3idx(EC, "d_ooL")
  vvLfile, dv_vvL = mmap3idx(EC, "d_vvL")
  

  #load dressed fock matrices
  SP = EC.space
  dfock = load2idx(EC, "df_mm")
  dfoo = dfock[SP['o'], SP['o']]
  dfov = dfock[SP['o'], SP['v']] #only internally dressed
  dfvo = dfock[SP['v'], SP['o']] 
  dfvv = dfock[SP['v'], SP['v']]

  #only for testing
  #@mtensor U1[a,i] := dfvo[a,i] 

  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nX = size(T_XX, 1)
  nL = size(v_ovL, 3)

  
  #@buffer buf(lenbuf) begin
  @mtensor tT2[a,b,i,j] := 2 * T2[a,b,i,j] - T2[b,a,i,j]          
  @mtensor Y_voL[b,j,L] := v_ovL[l,d,L] * tT2[d,b,l,j]  
  
  @mtensor tU2[a,b,i,j] := 2 * U2[a,b,i,j] - U2[b,a,i,j] 
  @mtensor tY_voL[b,j,L] := v_ovL[l,d,L] * tU2[d,b,l,j] 
  
  @mtensor A[k,bX,c,L] := dv_vvL[a,c,L] * bU_obXv[k,bX,a]
  @mtensor v_bXbXL[bX,bX',L] := A[k,bX,c,L] * bU_vobX[c,k,bX']
  
  @mtensor dv_bXbXL[bX,bX',L] := v_bXbXL[bX,bX',L] 
  @mtensor A[i,bX,k,bX'] := bU_obXv[i,bX,c] * bU_vobX[c,k,bX']
  @mtensor dv_bXbXL[bX,bX',L] -= dv_ooL[k,i,L] * A[i,bX,k,bX']
 
  @mtensor bv_ooL[k,m,L] := v_ovL[k,e,L] * U1[e,m]
 
  @mtensor A[i,bX,j,X'] := U_voX[a,j,X'] * bU_obXv[i,bX,a]
  @mtensor bv_bXXL[bX,X',L] := bv_ooL[j,i,L] * A[i,bX,j,X']
  
  @mtensor A[m,i,X',L] := v_ovL[m,c,L] * U_voX[c,i,X']
  @mtensor B[i,bX,m] := U1[a,m] * bU_obXv[i,bX,a]
  @mtensor bv_bXXL[bX,X',L] += A[m,i,X',L] * B[i,bX,m]
  
  @mtensor A[j,bY,d,L] := dv_vvL[b,d,L] * bU_obXv[j,bY,b]
  @mtensor dv_bXXL[bY,Y',L] := - A[j,bY,d,L] * U_voX[d,j,Y'] 
  @mtensor A[j,bY,l,Y'] := U_voX[b,l,Y'] * bU_obXv[j,bY,b]
  @mtensor dv_bXXL[bY,Y',L] += A[j,bY,l,Y'] * dv_ooL[l,j,L]
  
  @mtensor dx_vv[a,c] := dfvv[a,c] 
  @mtensor dx_vv[a,c] -= 0.5 * Y_voL[a,k,L] * v_ovL[k,c,L]
  
  @mtensor dx_oo[k,i] := dfoo[k,i] 
  @mtensor dx_oo[k,i] += 0.5 * Y_voL[c,i,L] * v_ovL[k,c,L]
  
  @mtensor bv_L[L] := v_ovL[m,e,L] * U1[e,m]
  @mtensor bv_voL[a,m,L] := dv_vvL[a,e,L] * U1[e,m]

  @mtensor bv_vv[a,d] := 2 * bv_L[L] * dv_vvL[a,d,L] 
  @mtensor bv_vv[a,d] -= bv_voL[a,m,L] * v_ovL[m,d,L]

  @mtensor bv_oo[k,i] := 2 * bv_L[L] * dv_ooL[k,i,L] 
  @mtensor bv_oo[k,i] -= bv_ooL[k,m,L] * dv_ooL[m,i,L]

  @mtensor bv_vo[a,i] := 2 * bv_L[L] * dv_voL[a,i,L] 
  @mtensor bv_vo[a,i] -= bv_voL[a,m,L] * dv_ooL[m,i,L]

  @mtensor bv_ov[k,b] := 2 * bv_L[L] * v_ovL[k,b,L] 
  @mtensor bv_ov[k,b] -= bv_ooL[k,m,L] *  v_ovL[m,b,L]

  @mtensor A[a,k,c,i] := dv_vvL[a,c,L] * dv_ooL[k,i,L]
  @mtensor dv_vobX[a,i,bX] := A[a,k,c,i] * bU_vobX[c,k,bX]

  @mtensor A[m,k,c,i] := v_ovL[m,c,L] * dv_ooL[k,i,L]
  @mtensor B[m,i,X] := A[m,k,c,i] * U_voX[c,k,X]
  @mtensor bv_voX[a,i,X] := B[m,i,X] * U1[a,m]
  @mtensor A[a,k,c,i] := bv_ooL[k,i,L] * dv_vvL[a,c,L]
  @mtensor bv_voX[a,i,X] -= A[a,k,c,i] * U_voX[c,k,X]



  @mtensor AU1[a,i] := - dfoo[m,i] * U1[a,m]
  @mtensor AU1[a,i] += dfvv[a,e] * U1[e,i]
  @mtensor AU1[a,i] += bv_vo[a,i] 
  @mtensor AU1[a,i] += bv_ov[j,b] * tT2[a,b,i,j]
  @mtensor A[m,i]  := Y_voL[b,i,L] * v_ovL[m,b,L] 
  @mtensor AU1[a,i] -= A[m,i] * U1[a,m]
  @mtensor AU1[a,i] -= Y_voL[a,j,L] * bv_ooL[j,i,L]
  @mtensor AU1[a,i] += dfov[m,e] * tU2[a,e,i,m] 
  @mtensor AU1[a,i] += dv_vvL[a,f,L] * tY_voL[f,i,L] 
  @mtensor AU1[a,i] += dv_ooL[n,i,L] * tY_voL[a,n,L] 
  
  @mtensor A[a,i,L] := dv_vvL[a,e,L] * U1[e,i]
  @mtensor A[a,i,L] -= dv_ooL[m,i,L] * U1[a,m]
  @mtensor A[a,i,L] += tY_voL[a,i,L]
  @mtensor B[b,j,L] := dv_voL[b,j,L] + Y_voL[b,j,L]
  @mtensor Q[a,b,i,j] := A[a,i,L] * B[b,j,L] 
  @mtensor A[bX,Y',L] := bv_bXXL[bX,X',L] * T_XX[X',Y']
  @mtensor B[bX,bY] := A[bX,Y',L] * dv_bXXL[bY,Y',L]
  @mtensor A[bX,bY',L] := 0.5 * dv_bXbXL[bX,bX',L] * U_bXbX[bX',bY']
  @mtensor B[bX,bY] += A[bX,bY',L] * dv_bXbXL[bY,bY',L]
  @mtensor C[bX,b,j] := B[bX,bY] * bU_vobX[b,j,bY]
  @mtensor Q[a,b,i,j] += C[bX,b,j] * bU_vobX[a,i,bX]
  @mtensor A[a,d] := - dfov[m,d] * U1[a,m]  
  @mtensor A[a,d] += bv_vv[a,d]
  @mtensor A[a,d] -= 0.5 * tY_voL[a,k,L] * v_ovL[k,d,L]
  @mtensor Q[a,b,i,j] += A[a,d] * T2[d,b,i,j]
  @mtensor A[k,i] := - dfov[k,e] * U1[e,i]
  @mtensor A[k,i] -= bv_oo[k,i]
  @mtensor A[k,i] -= 0.5 * tY_voL[e,i,L] * v_ovL[k,e,L]
  @mtensor Q[a,b,i,j] += A[k,i] * T2[a,b,k,j]
  @mtensor Q[a,b,i,j] += dx_vv[a,e] * U2[e,b,i,j]
  @mtensor Q[a,b,i,j] -= dx_oo[m,i]* U2[a,b,m,j]
  @mtensor A[bX,b,j]  := bU_obXv[m,bX,e] * U2[e,b,m,j]
  @mtensor Q[a,b,i,j] -= dv_vobX[a,i,bX] * A[bX,b,j]
  @mtensor A[X,b,j] := U_oXv[k,X,c] * T2[c,b,k,j]
  @mtensor Q[a,b,i,j] += bv_voX[a,i,X] * A[X,b,j]

  @mtensor AU2[a,b,i,j] := Q[a,b,i,j] + Q[b,a,j,i]
  #end
  
  close(voLfile)
  close(ovLfile)
  close(ooLfile)
  close(vvLfile)
  return AU1, AU2
end
