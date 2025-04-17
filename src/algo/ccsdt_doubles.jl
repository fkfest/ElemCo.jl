function ccsdt_doubles!(EC::ECInfo, R2a, R2b, R2ab, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
d_vvvv = load4idx(EC,"d_vvvv")
@mtensoropt begin
X[c,d,i,j] := d_vvvv[c,d,a,b] * T2a[a,b,i,j]
R2a[c,d,i,j] += 0.5 * X[c,d,i,j]
R2a[c,d,i,j] -= 0.5 * X[d,c,i,j]
end
d_vvvv = nothing
d_VVVV = load4idx(EC,"d_VVVV")
@mtensoropt begin
X[C,D,I,J] := d_VVVV[D,C,B,A] * T2b[A,B,I,J]
R2b[C,D,I,J] += 0.5 * X[C,D,I,J]
R2b[C,D,I,J] -= 0.5 * X[D,C,I,J]
end
d_VVVV = nothing
d_vVvV = load4idx(EC,"d_vVvV")
@mtensoropt R2ab[b,B,i,I] += d_vVvV[b,B,a,A] * T2ab[a,A,i,I]
d_vVvV = nothing
d_vovv = load4idx(EC,"d_vovv")
@mtensoropt R2ab[c,A,j,I] += d_vovv[c,i,b,a] * T3aab[b,a,A,j,i,I]
@mtensoropt begin
X[c,d,j,k] := d_vovv[c,i,b,a] * T3a[d,b,a,j,k,i]
R2a[c,d,j,k] -= X[c,d,j,k]
R2a[d,c,j,k] += X[c,d,j,k]
end
d_vovv = nothing
d_VOVV = load4idx(EC,"d_VOVV")
@mtensoropt R2ab[a,C,i,J] += d_VOVV[C,I,B,A] * T3abb[a,B,A,i,J,I]
@mtensoropt begin
X[C,D,J,K] := d_VOVV[C,I,B,A] * T3b[D,B,A,J,K,I]
R2b[C,D,J,K] -= X[C,D,J,K]
R2b[D,C,J,K] += X[C,D,J,K]
end
d_VOVV = nothing
d_vOvV = load4idx(EC,"d_vOvV")
@mtensoropt R2ab[b,B,i,J] += d_vOvV[b,I,a,A] * T3abb[a,B,A,i,J,I]
@mtensoropt begin
X[b,c,i,j] := d_vOvV[b,I,a,A] * T3aab[c,a,A,i,j,I]
R2a[b,c,i,j] -= X[b,c,i,j]
R2a[c,b,i,j] += X[b,c,i,j]
end
d_vOvV = nothing
d_vvoo = load4idx(EC,"d_vvoo")
@mtensoropt R2a[a,b,i,j] -= d_vvoo[b,a,i,j]
@mtensoropt R2a[a,b,i,j] += d_vvoo[a,b,i,j]
d_vvoo = nothing
d_VVOO = load4idx(EC,"d_VVOO")
@mtensoropt R2b[A,B,I,J] -= d_VVOO[A,B,J,I]
@mtensoropt R2b[A,B,I,J] += d_VVOO[B,A,J,I]
d_VVOO = nothing
d_vVoO = load4idx(EC,"d_vVoO")
@mtensoropt R2ab[a,A,i,I] += d_vVoO[a,A,i,I]
d_vVoO = nothing
d_vovo = load4idx(EC,"d_vovo")
@mtensoropt R2ab[b,A,j,I] -= d_vovo[b,i,a,j] * T2ab[a,A,i,I]
@mtensoropt begin
X[b,c,j,k] := d_vovo[b,i,a,j] * T2a[c,a,k,i]
R2a[b,c,j,k] -= X[b,c,j,k]
R2a[c,b,j,k] += X[b,c,j,k]
R2a[b,c,k,j] += X[b,c,j,k]
R2a[c,b,k,j] -= X[b,c,j,k]
end
d_vovo = nothing
d_VOVO = load4idx(EC,"d_VOVO")
@mtensoropt R2ab[a,B,i,J] -= d_VOVO[B,I,A,J] * T2ab[a,A,i,I]
@mtensoropt begin
X[B,C,J,K] := d_VOVO[B,I,A,J] * T2b[C,A,K,I]
R2b[B,C,J,K] -= X[B,C,J,K]
R2b[C,B,J,K] += X[B,C,J,K]
R2b[B,C,K,J] += X[B,C,J,K]
R2b[C,B,K,J] -= X[B,C,J,K]
end
d_VOVO = nothing
d_vOvO = load4idx(EC,"d_vOvO")
@mtensoropt R2ab[b,A,i,J] -= d_vOvO[b,I,a,J] * T2ab[a,A,i,I]
d_vOvO = nothing
d_voov = load4idx(EC,"d_voov")
@mtensoropt R2ab[b,A,j,I] += d_voov[b,i,j,a] * T2ab[a,A,i,I]
@mtensoropt begin
X[b,c,j,k] := d_voov[b,i,j,a] * T2a[c,a,k,i]
R2a[b,c,j,k] += X[b,c,j,k]
R2a[c,b,j,k] -= X[b,c,j,k]
R2a[b,c,k,j] -= X[b,c,j,k]
R2a[c,b,k,j] += X[b,c,j,k]
end
d_voov = nothing
d_VOOV = load4idx(EC,"d_VOOV")
@mtensoropt R2ab[a,B,i,J] += d_VOOV[B,I,J,A] * T2ab[a,A,i,I]
@mtensoropt begin
X[B,C,J,K] := d_VOOV[B,I,J,A] * T2b[C,A,K,I]
R2b[B,C,J,K] += X[B,C,J,K]
R2b[C,B,J,K] -= X[B,C,J,K]
R2b[B,C,K,J] -= X[B,C,J,K]
R2b[C,B,K,J] += X[B,C,J,K]
end
d_VOOV = nothing
d_vOoV = load4idx(EC,"d_vOoV")
@mtensoropt R2ab[a,B,i,J] += d_vOoV[a,I,i,A] * T2b[B,A,J,I]
@mtensoropt begin
X[a,b,i,j] := d_vOoV[a,I,i,A] * T2ab[b,A,j,I]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
R2a[a,b,j,i] -= X[a,b,i,j]
R2a[b,a,j,i] += X[a,b,i,j]
end
d_vOoV = nothing
d_oooo = load4idx(EC,"d_oooo")
@mtensoropt begin
X[a,b,k,l] := d_oooo[j,i,l,k] * T2a[a,b,i,j]
R2a[a,b,k,l] += 0.5 * X[a,b,k,l]
R2a[a,b,k,l] -= 0.5 * X[b,a,k,l]
end
d_oooo = nothing
d_OOOO = load4idx(EC,"d_OOOO")
@mtensoropt begin
X[A,B,K,L] := d_OOOO[J,I,L,K] * T2b[A,B,I,J]
R2b[A,B,K,L] += 0.5 * X[A,B,K,L]
R2b[A,B,K,L] -= 0.5 * X[B,A,K,L]
end
d_OOOO = nothing
d_oOoO = load4idx(EC,"d_oOoO")
@mtensoropt R2ab[a,A,j,J] += d_oOoO[i,I,j,J] * T2ab[a,A,i,I]
d_oOoO = nothing
d_oovo = load4idx(EC,"d_oovo")
@mtensoropt R2ab[b,A,k,I] -= d_oovo[j,i,a,k] * T3aab[b,a,A,i,j,I]
@mtensoropt begin
X[b,c,k,l] := d_oovo[j,i,a,k] * T3a[b,c,a,l,i,j]
R2a[b,c,k,l] += X[b,c,k,l]
R2a[b,c,l,k] -= X[b,c,k,l]
end
d_oovo = nothing
d_OOVO = load4idx(EC,"d_OOVO")
@mtensoropt R2ab[a,B,i,K] -= d_OOVO[J,I,A,K] * T3abb[a,B,A,i,I,J]
@mtensoropt begin
X[B,C,K,L] := d_OOVO[J,I,A,K] * T3b[B,C,A,L,I,J]
R2b[B,C,K,L] += X[B,C,K,L]
R2b[B,C,L,K] -= X[B,C,K,L]
end
d_OOVO = nothing
d_oOvO = load4idx(EC,"d_oOvO")
@mtensoropt R2ab[b,A,j,J] -= d_oOvO[i,I,a,J] * T3aab[b,a,A,j,i,I]
@mtensoropt begin
X[A,B,J,K] := d_oOvO[i,I,a,J] * T3abb[a,A,B,i,K,I]
R2b[A,B,J,K] += X[A,B,J,K]
R2b[A,B,K,J] -= X[A,B,J,K]
end
d_oOvO = nothing
d_oOoV = load4idx(EC,"d_oOoV")
@mtensoropt R2ab[a,B,j,J] -= d_oOoV[i,I,j,A] * T3abb[a,B,A,i,J,I]
@mtensoropt begin
X[a,b,j,k] := d_oOoV[i,I,j,A] * T3aab[a,b,A,k,i,I]
R2a[a,b,j,k] += X[a,b,j,k]
R2a[a,b,k,j] -= X[a,b,j,k]
end
d_oOoV = nothing
d_oVvO = load4idx(EC,"d_oVvO")
@mtensoropt R2ab[b,A,j,I] += d_oVvO[i,A,a,I] * T2a[b,a,j,i]
@mtensoropt begin
X[A,B,I,J] := d_oVvO[i,A,a,I] * T2ab[a,B,i,J]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
R2b[A,B,J,I] -= X[A,B,I,J]
R2b[B,A,J,I] += X[A,B,I,J]
end
d_oVvO = nothing
d_oVoV = load4idx(EC,"d_oVoV")
@mtensoropt R2ab[a,B,j,I] -= d_oVoV[i,B,j,A] * T2ab[a,A,i,I]
d_oVoV = nothing
oovv = ints2(EC,"oovv")
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,k=>x,j=>x,b=>10*x,c=>10*x) R2ab[a,A,i,I] -= oovv[k,j,b,c] * T2a[a,b,i,j] * T2ab[c,A,k,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,k=>x,j=>x,c=>10*x,b=>10*x) R2ab[a,A,i,I] -= oovv[k,j,c,b] * T2a[c,b,i,j] * T2ab[a,A,k,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,k=>x,j=>x,c=>10*x,b=>10*x) R2ab[a,A,i,I] -= oovv[k,j,c,b] * T2a[b,a,j,k] * T2ab[c,A,i,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,k=>x,j=>x,c=>10*x,b=>10*x) R2ab[a,A,i,I] += oovv[k,j,c,b] * T2a[a,b,i,j] * T2ab[c,A,k,I]
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,j=>x,i=>x,a=>10*x,b=>10*x) begin
X[A,B,I,J] := oovv[j,i,a,b] * T2ab[a,A,i,I] * T2ab[b,B,j,J]
R2b[A,B,I,J] -= X[A,B,I,J]
R2b[B,A,I,J] += X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,j=>x,i=>x,b=>10*x,a=>10*x) begin
X[A,B,I,J] := oovv[j,i,b,a] * T2ab[a,A,i,I] * T2ab[b,B,j,J]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2a[a,b,k,l] * T2a[c,d,i,j]
R2a[a,b,i,j] += 0.25 * X[a,b,i,j]
R2a[a,b,i,j] -= 0.25 * X[b,a,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,i,j] := oovv[l,k,c,d] * T2a[a,c,i,k] * T2a[b,d,j,l]
R2a[a,b,i,j] -= X[a,b,i,j]
R2a[b,a,i,j] += X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2a[d,c,i,k] * T2a[a,b,j,l]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[a,b,j,i] -= X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2a[c,a,k,l] * T2a[b,d,i,j]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2a[a,c,i,k] * T2a[b,d,j,l]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
oovv = nothing
OOVV = ints2(EC,"OOVV")
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) R2ab[a,A,i,I] -= OOVV[K,J,C,B] * T2b[C,B,I,J] * T2ab[a,A,i,K]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) R2ab[a,A,i,I] -= OOVV[K,J,C,B] * T2b[B,A,J,K] * T2ab[a,C,i,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,K=>x,J=>x,B=>10*x,C=>10*x) R2ab[a,A,i,I] -= OOVV[K,J,B,C] * T2ab[a,B,i,J] * T2b[A,C,I,K]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) R2ab[a,A,i,I] += OOVV[K,J,C,B] * T2ab[a,B,i,J] * T2b[A,C,I,K]
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[A,B,I,J] := OOVV[L,K,D,C] * T2b[A,B,K,L] * T2b[C,D,I,J]
R2b[A,B,I,J] += 0.25 * X[A,B,I,J]
R2b[A,B,I,J] -= 0.25 * X[B,A,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,L=>x,K=>x,C=>10*x,D=>10*x) begin
X[A,B,I,J] := OOVV[L,K,C,D] * T2b[A,C,I,K] * T2b[B,D,J,L]
R2b[A,B,I,J] -= X[A,B,I,J]
R2b[B,A,I,J] += X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[A,B,I,J] := OOVV[L,K,D,C] * T2b[D,C,I,K] * T2b[A,B,J,L]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[A,B,J,I] -= X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[A,B,I,J] := OOVV[L,K,D,C] * T2b[C,A,K,L] * T2b[B,D,I,J]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[A,B,I,J] := OOVV[L,K,D,C] * T2b[A,C,I,K] * T2b[B,D,J,L]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,J=>x,I=>x,A=>10*x,B=>10*x) begin
X[a,b,i,j] := OOVV[J,I,A,B] * T2ab[a,A,i,I] * T2ab[b,B,j,J]
R2a[a,b,i,j] -= X[a,b,i,j]
R2a[b,a,i,j] += X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,J=>x,I=>x,B=>10*x,A=>10*x) begin
X[a,b,i,j] := OOVV[J,I,B,A] * T2ab[a,A,i,I] * T2ab[b,B,j,J]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
OOVV = nothing
oOvV = ints2(EC,"oOvV")
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] += oOvV[j,J,b,B] * T2ab[a,A,j,J] * T2ab[b,B,i,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] -= oOvV[j,J,b,B] * T2ab[b,B,j,I] * T2ab[a,A,i,J]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] += oOvV[j,J,b,B] * T2ab[b,A,i,J] * T2ab[a,B,j,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] -= oOvV[j,J,b,B] * T2ab[b,A,j,J] * T2ab[a,B,i,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] -= oOvV[j,J,b,B] * T2ab[b,B,i,J] * T2ab[a,A,j,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] -= oOvV[j,J,b,B] * T2ab[a,B,j,J] * T2ab[b,A,i,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] += oOvV[j,J,b,B] * T2ab[a,B,i,J] * T2ab[b,A,j,I]
@mtensoropt (a=>10*x,A=>10*x,i=>x,I=>x,j=>x,J=>x,b=>10*x,B=>10*x) R2ab[a,A,i,I] += oOvV[j,J,b,B] * T2a[a,b,i,j] * T2b[A,B,I,J]
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,i=>x,K=>x,a=>10*x,C=>10*x) begin
X[A,B,I,J] := oOvV[i,K,a,C] * T2ab[a,C,i,I] * T2b[A,B,J,K]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[A,B,J,I] -= X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,i=>x,K=>x,a=>10*x,C=>10*x) begin
X[A,B,I,J] := oOvV[i,K,a,C] * T2ab[a,A,i,K] * T2b[B,C,I,J]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,i=>x,K=>x,a=>10*x,C=>10*x) begin
X[A,B,I,J] := oOvV[i,K,a,C] * T2b[A,C,I,K] * T2ab[a,B,i,J]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,I=>x,J=>x,i=>x,K=>x,a=>10*x,C=>10*x) begin
X[A,B,I,J] := oOvV[i,K,a,C] * T2ab[a,A,i,I] * T2b[B,C,J,K]
R2b[A,B,I,J] += X[A,B,I,J]
R2b[B,A,I,J] -= X[A,B,I,J]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,k=>x,I=>x,c=>10*x,A=>10*x) begin
X[a,b,i,j] := oOvV[k,I,c,A] * T2ab[c,A,i,I] * T2a[a,b,j,k]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[a,b,j,i] -= X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,k=>x,I=>x,c=>10*x,A=>10*x) begin
X[a,b,i,j] := oOvV[k,I,c,A] * T2ab[a,A,k,I] * T2a[b,c,i,j]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,k=>x,I=>x,c=>10*x,A=>10*x) begin
X[a,b,i,j] := oOvV[k,I,c,A] * T2ab[a,A,i,I] * T2a[b,c,j,k]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,k=>x,I=>x,c=>10*x,A=>10*x) begin
X[a,b,i,j] := oOvV[k,I,c,A] * T2a[a,c,i,k] * T2ab[b,A,j,I]
R2a[a,b,i,j] += X[a,b,i,j]
R2a[b,a,i,j] -= X[a,b,i,j]
end
oOvV = nothing
d_oVvV = load4idx(EC,"d_oVvV")
@mtensoropt R2ab[b,B,j,I] += d_oVvV[i,B,a,A] * T3aab[b,a,A,j,i,I]
@mtensoropt begin
X[B,C,I,J] := d_oVvV[i,B,a,A] * T3abb[a,C,A,i,I,J]
R2b[B,C,I,J] -= X[B,C,I,J]
R2b[C,B,I,J] += X[B,C,I,J]
end
d_oVvV = nothing
@mtensoropt R2a[b,c,j,k] += fia[i,a] * T3a[b,c,a,j,k,i]
@mtensoropt R2a[a,b,i,j] += fIA[I,A] * T3aab[a,b,A,i,j,I]
@mtensoropt begin
X[a,b,j,k] := fij[i,j] * T2a[a,b,k,i]
R2a[a,b,j,k] += X[a,b,j,k]
R2a[a,b,k,j] -= X[a,b,j,k]
end
@mtensoropt begin
X[b,c,i,j] := fab[b,a] * T2a[c,a,i,j]
R2a[b,c,i,j] -= X[b,c,i,j]
R2a[c,b,i,j] += X[b,c,i,j]
end
@mtensoropt R2b[A,B,I,J] += fia[i,a] * T3abb[a,A,B,i,I,J]
@mtensoropt R2b[B,C,J,K] += fIA[I,A] * T3b[B,C,A,J,K,I]
@mtensoropt begin
X[A,B,J,K] := fIJ[I,J] * T2b[A,B,K,I]
R2b[A,B,J,K] += X[A,B,J,K]
R2b[A,B,K,J] -= X[A,B,J,K]
end
@mtensoropt begin
X[B,C,I,J] := fAB[B,A] * T2b[C,A,I,J]
R2b[B,C,I,J] -= X[B,C,I,J]
R2b[C,B,I,J] += X[B,C,I,J]
end
@mtensoropt R2ab[b,A,j,I] += fia[i,a] * T3aab[b,a,A,j,i,I]
@mtensoropt R2ab[a,B,i,J] += fIA[I,A] * T3abb[a,B,A,i,J,I]
@mtensoropt R2ab[a,A,j,I] -= fij[i,j] * T2ab[a,A,i,I]
@mtensoropt R2ab[b,A,i,I] += fab[b,a] * T2ab[a,A,i,I]
@mtensoropt R2ab[a,B,i,I] += fAB[B,A] * T2ab[a,A,i,I]
@mtensoropt R2ab[a,A,i,J] -= fIJ[I,J] * T2ab[a,A,i,I]
end

function ccsdt_doubles!(EC::ECInfo, R2, T2, T3, fij, fab, fai, fia)
#bracd
d_vvvv = load4idx(EC,"d_vvvv")
@mtensoropt R2[c,d,i,j] += d_vvvv[c,d,a,b] * T2[a,b,i,j]
d_vvvv = nothing
d_vovv = load4idx(EC,"d_vovv")
@mtensoropt begin
X[c,d,j,k] := d_vovv[c,i,b,a] * T3[d,a,b,i,k,j]
R2[c,d,j,k] -= X[c,d,j,k]
R2[d,c,j,k] -= X[c,d,k,j]
end
@mtensoropt begin
X[c,d,j,k] := d_vovv[c,i,a,b] * T3[a,b,d,i,j,k]
R2[c,d,j,k] -= X[c,d,j,k]
R2[d,c,j,k] -= X[c,d,k,j]
end
@mtensoropt begin
X[c,d,j,k] := d_vovv[c,i,b,a] * T3[a,b,d,i,j,k]
R2[c,d,j,k] += 2 * X[c,d,j,k]
R2[d,c,j,k] += 2 * X[c,d,k,j]
end
d_vovv = nothing
d_vvoo = load4idx(EC,"d_vvoo")
@mtensoropt R2[a,b,i,j] += d_vvoo[a,b,i,j]
d_vvoo = nothing
d_vovo = load4idx(EC,"d_vovo")
@mtensoropt begin
X[c,b,j,k] := d_vovo[b,i,a,j] * T2[c,a,i,k]
R2[c,b,j,k] -= X[c,b,j,k]
R2[b,c,k,j] -= X[c,b,j,k]
end
@mtensoropt begin
X[b,c,j,k] := d_vovo[b,i,a,j] * T2[a,c,i,k]
R2[b,c,j,k] -= X[b,c,j,k]
R2[c,b,k,j] -= X[b,c,j,k]
end
d_vovo = nothing
d_voov = load4idx(EC,"d_voov")
@mtensoropt begin
X[b,c,j,k] := d_voov[b,i,j,a] * T2[c,a,i,k]
R2[b,c,j,k] -= X[b,c,j,k]
R2[c,b,k,j] -= X[b,c,j,k]
end
@mtensoropt begin
X[b,c,j,k] := d_voov[b,i,j,a] * T2[a,c,i,k]
R2[b,c,j,k] += 2 * X[b,c,j,k]
R2[c,b,k,j] += 2 * X[b,c,j,k]
end
d_voov = nothing
d_oooo = load4idx(EC,"d_oooo")
@mtensoropt R2[a,b,k,l] += d_oooo[j,i,l,k] * T2[a,b,i,j]
d_oooo = nothing
d_oovo = load4idx(EC,"d_oovo")
@mtensoropt begin
X[b,c,k,l] := d_oovo[j,i,a,k] * T3[b,c,a,i,j,l]
R2[b,c,k,l] += X[b,c,k,l]
R2[b,c,l,k] += X[c,b,k,l]
end
@mtensoropt begin
X[b,c,k,l] := d_oovo[j,i,a,k] * T3[a,b,c,i,j,l]
R2[b,c,k,l] += X[b,c,k,l]
R2[b,c,l,k] += X[c,b,k,l]
end
@mtensoropt begin
X[b,c,k,l] := d_oovo[j,i,a,k] * T3[b,a,c,i,j,l]
R2[b,c,k,l] -= 2 * X[b,c,k,l]
R2[b,c,l,k] -= 2 * X[c,b,k,l]
end
d_oovo = nothing
oovv = ints2(EC,"oovv")
@mtensoropt (a=>10*x,b=>10*x,j=>x,i=>x,l=>x,k=>x,c=>10*x,d=>10*x) R2[a,b,j,i] += oovv[l,k,c,d] * T2[a,c,k,i] * T2[d,b,j,l]
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,i,j] := oovv[l,k,c,d] * T2[a,c,k,i] * T2[d,b,l,j]
R2[a,b,i,j] += X[a,b,i,j]
R2[b,a,j,i] += X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) R2[a,b,i,j] += oovv[l,k,d,c] * T2[a,b,k,l] * T2[c,d,i,j]
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) R2[a,b,i,j] += oovv[l,k,d,c] * T2[a,c,k,i] * T2[b,d,l,j]
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,i,j] := oovv[l,k,c,d] * T2[c,d,k,i] * T2[a,b,l,j]
R2[a,b,i,j] += X[a,b,i,j]
R2[a,b,j,i] += X[b,a,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2[a,c,k,l] * T2[d,b,i,j]
R2[a,b,i,j] += X[a,b,i,j]
R2[b,a,i,j] += X[a,b,j,i]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,c=>10*x,d=>10*x) R2[a,b,i,j] -= 2 * oovv[l,k,c,d] * T2[c,a,k,i] * T2[d,b,l,j]
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2[c,a,k,i] * T2[b,d,l,j]
R2[a,b,i,j] -= 2 * X[a,b,i,j]
R2[b,a,j,i] -= 2 * X[a,b,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2[c,d,k,i] * T2[a,b,l,j]
R2[a,b,i,j] -= 2 * X[a,b,i,j]
R2[a,b,j,i] -= 2 * X[b,a,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,i,j] := oovv[l,k,d,c] * T2[c,a,k,l] * T2[d,b,i,j]
R2[a,b,i,j] -= 2 * X[a,b,i,j]
R2[b,a,i,j] -= 2 * X[a,b,j,i]
end
@mtensoropt (a=>10*x,b=>10*x,i=>x,j=>x,l=>x,k=>x,d=>10*x,c=>10*x) R2[a,b,i,j] += 4 * oovv[l,k,d,c] * T2[c,a,k,i] * T2[d,b,l,j]
oovv = nothing
@mtensoropt R2[b,c,j,k] += 2 * fia[i,a] * T3[a,b,c,i,j,k]
@mtensoropt begin
X[a,b,j,k] := fij[i,j] * T2[a,b,i,k]
R2[a,b,j,k] -= X[a,b,j,k]
R2[a,b,k,j] -= X[b,a,j,k]
end
@mtensoropt begin
X[b,c,j,k] := fia[i,a] * T3[b,a,c,i,j,k]
R2[b,c,j,k] -= X[b,c,j,k]
R2[b,c,j,k] -= X[c,b,k,j]
end
@mtensoropt begin
X[b,c,i,j] := fab[b,a] * T2[a,c,i,j]
R2[b,c,i,j] += X[b,c,i,j]
R2[c,b,i,j] += X[b,c,j,i]
end
end
