"""
    calc_R2_df_cis_pert_d(EC::ECInfo, U1)

Calculates doubles residuals for CIS(D) with dressed integrals.

"""
function calc_R2_df_cis_pert_d(EC::ECInfo, U1)
  t1 = time_ns()
  SP = EC.space

  #attention, dressed integrals are used on purpose! (then it is closer to CC2 than to CIS(D))
  voLfile, dv_voL = mmap3idx(EC, "d_voL")
  ooLfile, dv_ooL = mmap3idx(EC, "d_ooL")
  vvLfile, dv_vvL = mmap3idx(EC, "d_vvL")

  @mtensor A[a,i,L] := dv_vvL[a,e,L] * U1[e,i]
  @mtensor A[a,i,L] -= dv_ooL[m,i,L] * U1[a,m]
  @mtensor R2[a,b,i,j] := A[a,i,L] * dv_voL[b,j,L]
  t1 = print_time(EC, t1, "df_cis_pert_d: residuals", 3)

  close(voLfile)
  close(ooLfile)
  close(vvLfile)
  return R2
end
