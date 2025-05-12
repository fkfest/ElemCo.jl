"""
    cis_HU1(EC::ECInfo, U1)

Compute H times U1 for CIS.

``V^i_a = f_a^c U^i_c - f_k^i U^k_a + (2 v_{ak}^{ic} - v_{ak}^{ci}) U^k_c`` 
"""
function cis_HU1(EC::ECInfo, U1)
  t1 = time_ns()
  SP = EC.space

  f_mm = load2idx(EC, "f_mm")
  f_oo = f_mm[SP['o'], SP['o']]
  f_vv = f_mm[SP['v'], SP['v']]

  @mtensor V1[a,i] := f_vv[a,c] * U1[c,i] - f_oo[k,i] * U1[a,k]
  t1 = print_time(EC, t1, "cis_HU1: fock", 3)
  int2 = ints2(EC, "voov")
  @mtensor V1[a,i] += 2 * int2[a,k,i,c] * U1[c,k]
  int2 = ints2(EC, "vovo")
  @mtensor V1[a,i] -= int2[a,k,c,i] * U1[c,k]
  t1 = print_time(EC, t1, "cis_HU1: ints2", 3)
  return V1
end