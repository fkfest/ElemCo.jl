function ccsdt_singles!(EC::ECInfo, R1a, R1b, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
d_vovv = load4idx(EC,"d_vovv")
@tensoropt R1a[c,j] += d_vovv[c,i,b,a] * T2a[b,a,j,i]
d_vovv = nothing
d_VOVV = load4idx(EC,"d_VOVV")
@tensoropt R1b[C,J] += d_VOVV[C,I,B,A] * T2b[B,A,J,I]
d_VOVV = nothing
d_vOvV = load4idx(EC,"d_vOvV")
@tensoropt R1a[b,i] += d_vOvV[b,I,a,A] * T2ab[a,A,i,I]
d_vOvV = nothing
d_oovo = load4idx(EC,"d_oovo")
@tensoropt R1a[b,k] -= d_oovo[j,i,a,k] * T2a[b,a,i,j]
d_oovo = nothing
d_OOVO = load4idx(EC,"d_OOVO")
@tensoropt R1b[B,K] -= d_OOVO[J,I,A,K] * T2b[B,A,I,J]
d_OOVO = nothing
d_oOvO = load4idx(EC,"d_oOvO")
@tensoropt R1b[A,J] -= d_oOvO[i,I,a,J] * T2ab[a,A,i,I]
d_oOvO = nothing
d_oOoV = load4idx(EC,"d_oOoV")
@tensoropt R1a[a,j] -= d_oOoV[i,I,j,A] * T2ab[a,A,i,I]
d_oOoV = nothing
oovv = ints2(EC,"oovv")
@tensoropt R1b[A,I] += 0.5 * oovv[j,i,b,a] * T3aab[a,b,A,i,j,I]
@tensoropt R1a[c,k] += 0.5 * oovv[j,i,b,a] * T3a[c,a,b,k,i,j]
oovv = nothing
OOVV = ints2(EC,"OOVV")
@tensoropt R1b[C,K] += 0.5 * OOVV[J,I,B,A] * T3b[C,A,B,K,I,J]
@tensoropt R1a[a,i] += 0.5 * OOVV[J,I,B,A] * T3abb[a,A,B,i,I,J]
OOVV = nothing
oOvV = ints2(EC,"oOvV")
@tensoropt R1b[B,J] += oOvV[i,I,a,A] * T3abb[a,B,A,i,J,I]
@tensoropt R1a[b,j] += oOvV[i,I,a,A] * T3aab[b,a,A,j,i,I]
oOvV = nothing
d_oVvV = load4idx(EC,"d_oVvV")
@tensoropt R1b[B,I] += d_oVvV[i,B,a,A] * T2ab[a,A,i,I]
d_oVvV = nothing
@tensoropt R1a[b,j] += fia[i,a] * T2a[b,a,j,i]
@tensoropt R1a[a,i] += fIA[I,A] * T2ab[a,A,i,I]
@tensoropt R1a[a,i] += fai[a,i]
@tensoropt R1b[A,I] += fia[i,a] * T2ab[a,A,i,I]
@tensoropt R1b[B,J] += fIA[I,A] * T2b[B,A,J,I]
@tensoropt R1b[A,I] += fAI[A,I]
end

function ccsdt_singles!(EC::ECInfo, R1, T2, T3, fij, fab, fai, fia)
# bracs
d_vovv = load4idx(EC,"d_vovv")
@tensoropt R1[c,j] -= d_vovv[c,i,a,b] * T2[a,b,i,j]
@tensoropt R1[c,j] += 2 * d_vovv[c,i,b,a] * T2[a,b,i,j]
d_vovv = nothing
d_oovo = load4idx(EC,"d_oovo")
@tensoropt R1[b,k] += d_oovo[j,i,a,k] * T2[a,b,i,j]
@tensoropt R1[b,k] -= 2 * d_oovo[j,i,a,k] * T2[b,a,i,j]
d_oovo = nothing
oovv = ints2(EC,"oovv")
@tensoropt R1[c,k] += oovv[j,i,b,a] * T3[c,a,b,i,j,k]
@tensoropt R1[c,k] -= oovv[j,i,a,b] * T3[a,b,c,i,j,k]
@tensoropt R1[c,k] -= 2 * oovv[j,i,b,a] * T3[a,c,b,i,j,k]
@tensoropt R1[c,k] += 2 * oovv[j,i,b,a] * T3[a,b,c,i,j,k]
oovv = nothing
@tensoropt R1[b,j] += 2 * fia[i,a] * T2[a,b,i,j]
@tensoropt R1[b,j] -= fia[i,a] * T2[b,a,i,j]
@tensoropt R1[a,i] += fai[a,i]
end
