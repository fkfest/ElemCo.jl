function dcccsdt_triples!(EC::ECInfo, R3a, R3b, R3aab, R3abb, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
d_vvvv = load4idx(EC,"d_vvvv")
@mtensoropt begin
X[c,d,A,i,j,I] := d_vvvv[c,d,a,b] * T3aab[a,b,A,i,j,I]
R3aab[c,d,A,i,j,I] += 0.5 * X[c,d,A,i,j,I]
R3aab[c,d,A,i,j,I] -= 0.5 * X[d,c,A,i,j,I]
end
@mtensoropt begin
X[c,d,e,i,j,k] := d_vvvv[c,d,a,b] * T3a[e,a,b,i,j,k]
R3a[c,d,e,i,j,k] += 0.5 * X[c,d,e,i,j,k]
R3a[c,e,d,i,j,k] -= 0.5 * X[c,d,e,i,j,k]
R3a[c,d,e,i,j,k] -= 0.5 * X[d,c,e,i,j,k]
R3a[e,c,d,i,j,k] += 0.5 * X[c,d,e,i,j,k]
R3a[c,e,d,i,j,k] += 0.5 * X[d,c,e,i,j,k]
R3a[e,c,d,i,j,k] -= 0.5 * X[d,c,e,i,j,k]
end
d_vvvv = nothing
d_VVVV = load4idx(EC,"d_VVVV")
@mtensoropt begin
X[a,C,D,i,I,J] := d_VVVV[D,C,B,A] * T3abb[a,A,B,i,I,J]
R3abb[a,C,D,i,I,J] += 0.5 * X[a,C,D,i,I,J]
R3abb[a,C,D,i,I,J] -= 0.5 * X[a,D,C,i,I,J]
end
@mtensoropt begin
X[C,D,E,I,J,K] := d_VVVV[D,C,B,A] * T3b[E,A,B,I,J,K]
R3b[C,D,E,I,J,K] += 0.5 * X[C,D,E,I,J,K]
R3b[C,E,D,I,J,K] -= 0.5 * X[C,D,E,I,J,K]
R3b[C,D,E,I,J,K] -= 0.5 * X[D,C,E,I,J,K]
R3b[E,C,D,I,J,K] += 0.5 * X[C,D,E,I,J,K]
R3b[C,E,D,I,J,K] += 0.5 * X[D,C,E,I,J,K]
R3b[E,C,D,I,J,K] -= 0.5 * X[D,C,E,I,J,K]
end
d_VVVV = nothing
d_vVvV = load4idx(EC,"d_vVvV")
@mtensoropt begin
X[b,c,B,i,j,I] := d_vVvV[b,B,a,A] * T3aab[c,a,A,i,j,I]
R3aab[b,c,B,i,j,I] -= 0.5 * X[b,c,B,i,j,I]
R3aab[c,b,B,i,j,I] += 0.5 * X[b,c,B,i,j,I]
R3aab[b,c,B,i,j,I] += 0.5 * X[b,c,B,j,i,I]
R3aab[c,b,B,i,j,I] -= 0.5 * X[b,c,B,j,i,I]
end
@mtensoropt begin
X[b,B,C,i,I,J] := d_vVvV[b,B,a,A] * T3abb[a,C,A,i,I,J]
R3abb[b,B,C,i,I,J] -= 0.5 * X[b,B,C,i,I,J]
R3abb[b,C,B,i,I,J] += 0.5 * X[b,B,C,i,I,J]
R3abb[b,B,C,i,I,J] += 0.5 * X[b,B,C,i,J,I]
R3abb[b,C,B,i,I,J] -= 0.5 * X[b,B,C,i,J,I]
end
d_vVvV = nothing
d_vvvo = load4idx(EC,"d_vvvo")
@mtensoropt begin
X[b,c,A,i,j,I] := d_vvvo[c,b,a,i] * T2ab[a,A,j,I]
R3aab[b,c,A,i,j,I] += X[b,c,A,i,j,I]
R3aab[b,c,A,i,j,I] -= X[c,b,A,i,j,I]
R3aab[b,c,A,j,i,I] -= X[b,c,A,i,j,I]
R3aab[b,c,A,j,i,I] += X[c,b,A,i,j,I]
end
@mtensoropt begin
X[b,c,d,i,j,k] := d_vvvo[c,b,a,i] * T2a[d,a,j,k]
R3a[b,c,d,i,j,k] -= X[b,c,d,i,j,k]
R3a[b,d,c,i,j,k] += X[b,c,d,i,j,k]
R3a[b,c,d,i,j,k] += X[c,b,d,i,j,k]
R3a[d,b,c,i,j,k] -= X[b,c,d,i,j,k]
R3a[b,d,c,i,j,k] -= X[c,b,d,i,j,k]
R3a[d,b,c,i,j,k] += X[c,b,d,i,j,k]
R3a[b,c,d,j,i,k] += X[b,c,d,i,j,k]
R3a[b,d,c,j,i,k] -= X[b,c,d,i,j,k]
R3a[b,c,d,j,i,k] -= X[c,b,d,i,j,k]
R3a[d,b,c,j,i,k] += X[b,c,d,i,j,k]
R3a[b,d,c,j,i,k] += X[c,b,d,i,j,k]
R3a[d,b,c,j,i,k] -= X[c,b,d,i,j,k]
R3a[b,c,d,j,k,i] -= X[b,c,d,i,j,k]
R3a[b,d,c,j,k,i] += X[b,c,d,i,j,k]
R3a[b,c,d,j,k,i] += X[c,b,d,i,j,k]
R3a[d,b,c,j,k,i] -= X[b,c,d,i,j,k]
R3a[b,d,c,j,k,i] -= X[c,b,d,i,j,k]
R3a[d,b,c,j,k,i] += X[c,b,d,i,j,k]
end
d_vvvo = nothing
d_VVVO = load4idx(EC,"d_VVVO")
@mtensoropt begin
X[a,B,C,i,I,J] := d_VVVO[C,B,A,I] * T2ab[a,A,i,J]
R3abb[a,B,C,i,I,J] += X[a,B,C,i,I,J]
R3abb[a,B,C,i,I,J] -= X[a,C,B,i,I,J]
R3abb[a,B,C,i,J,I] -= X[a,B,C,i,I,J]
R3abb[a,B,C,i,J,I] += X[a,C,B,i,I,J]
end
@mtensoropt begin
X[B,C,D,I,J,K] := d_VVVO[C,B,A,I] * T2b[D,A,J,K]
R3b[B,C,D,I,J,K] -= X[B,C,D,I,J,K]
R3b[B,D,C,I,J,K] += X[B,C,D,I,J,K]
R3b[B,C,D,I,J,K] += X[C,B,D,I,J,K]
R3b[D,B,C,I,J,K] -= X[B,C,D,I,J,K]
R3b[B,D,C,I,J,K] -= X[C,B,D,I,J,K]
R3b[D,B,C,I,J,K] += X[C,B,D,I,J,K]
R3b[B,C,D,J,I,K] += X[B,C,D,I,J,K]
R3b[B,D,C,J,I,K] -= X[B,C,D,I,J,K]
R3b[B,C,D,J,I,K] -= X[C,B,D,I,J,K]
R3b[D,B,C,J,I,K] += X[B,C,D,I,J,K]
R3b[B,D,C,J,I,K] += X[C,B,D,I,J,K]
R3b[D,B,C,J,I,K] -= X[C,B,D,I,J,K]
R3b[B,C,D,J,K,I] -= X[B,C,D,I,J,K]
R3b[B,D,C,J,K,I] += X[B,C,D,I,J,K]
R3b[B,C,D,J,K,I] += X[C,B,D,I,J,K]
R3b[D,B,C,J,K,I] -= X[B,C,D,I,J,K]
R3b[B,D,C,J,K,I] -= X[C,B,D,I,J,K]
R3b[D,B,C,J,K,I] += X[C,B,D,I,J,K]
end
d_VVVO = nothing
d_vVvO = load4idx(EC,"d_vVvO")
@mtensoropt begin
X[b,c,A,i,j,I] := d_vVvO[b,A,a,I] * T2a[c,a,i,j]
R3aab[b,c,A,i,j,I] -= X[b,c,A,i,j,I]
R3aab[c,b,A,i,j,I] += X[b,c,A,i,j,I]
end
@mtensoropt begin
X[b,A,B,i,I,J] := d_vVvO[b,A,a,I] * T2ab[a,B,i,J]
R3abb[b,A,B,i,I,J] += X[b,A,B,i,I,J]
R3abb[b,B,A,i,I,J] -= X[b,A,B,i,I,J]
R3abb[b,A,B,i,J,I] -= X[b,A,B,i,I,J]
R3abb[b,B,A,i,J,I] += X[b,A,B,i,I,J]
end
d_vVvO = nothing
d_vovv = load4idx(EC,"d_vovv")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := d_vovv[a,k,d,c] * T2ab[b,A,k,I] * T2a[c,d,i,j]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,j,i,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,j,i,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,A,i,j,I] := d_vovv[a,k,c,d] * T2a[b,c,i,k] * T2ab[d,A,j,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,A,i,j,I] := d_vovv[a,k,c,d] * T2ab[c,A,k,I] * T2a[b,d,i,j]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := d_vovv[a,k,d,c] * T2a[b,c,i,k] * T2ab[d,A,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := d_vovv[a,k,d,c] * T2ab[c,A,k,I] * T2a[b,d,i,j]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x,c=>10*x) begin
X[a,A,B,i,I,J] := d_vovv[a,j,b,c] * T2ab[b,A,j,I] * T2ab[c,B,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,c=>10*x,b=>10*x) begin
X[a,A,B,i,I,J] := d_vovv[a,j,c,b] * T2ab[b,A,j,I] * T2ab[c,B,i,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,e,d] * T2a[b,c,i,l] * T2a[d,e,j,k]
R3a[a,b,c,j,i,k] += 0.5 * X[a,b,c,j,i,k]
R3a[a,b,c,j,k,i] -= 0.5 * X[a,b,c,j,i,k]
R3a[b,a,c,j,i,k] -= 0.5 * X[a,b,c,j,i,k]
R3a[b,a,c,j,k,i] += 0.5 * X[a,b,c,j,i,k]
R3a[b,c,a,j,i,k] += 0.5 * X[a,b,c,j,i,k]
R3a[b,c,a,j,k,i] -= 0.5 * X[a,b,c,j,i,k]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,j,i,k]
R3a[a,b,c,j,k,i] += 0.5 * X[a,b,c,k,i,j]
R3a[b,a,c,i,j,k] += 0.5 * X[a,b,c,j,i,k]
R3a[b,a,c,j,k,i] -= 0.5 * X[a,b,c,k,i,j]
R3a[b,c,a,i,j,k] -= 0.5 * X[a,b,c,j,i,k]
R3a[b,c,a,j,k,i] += 0.5 * X[a,b,c,k,i,j]
R3a[a,b,c,i,j,k] += 0.5 * X[a,b,c,k,i,j]
R3a[a,b,c,j,i,k] -= 0.5 * X[a,b,c,k,i,j]
R3a[b,a,c,i,j,k] -= 0.5 * X[a,b,c,k,i,j]
R3a[b,a,c,j,i,k] += 0.5 * X[a,b,c,k,i,j]
R3a[b,c,a,i,j,k] += 0.5 * X[a,b,c,k,i,j]
R3a[b,c,a,j,i,k] -= 0.5 * X[a,b,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,d,e] * T2a[b,d,i,l] * T2a[c,e,j,k]
R3a[a,b,c,j,i,k] += X[a,b,c,j,i,k]
R3a[a,c,b,j,i,k] -= X[a,b,c,j,i,k]
R3a[a,b,c,j,k,i] -= X[a,b,c,j,i,k]
R3a[a,c,b,j,k,i] += X[a,b,c,j,i,k]
R3a[b,a,c,j,i,k] -= X[a,b,c,j,i,k]
R3a[c,a,b,j,i,k] += X[a,b,c,j,i,k]
R3a[b,a,c,j,k,i] += X[a,b,c,j,i,k]
R3a[c,a,b,j,k,i] -= X[a,b,c,j,i,k]
R3a[b,c,a,j,i,k] += X[a,b,c,j,i,k]
R3a[c,b,a,j,i,k] -= X[a,b,c,j,i,k]
R3a[b,c,a,j,k,i] -= X[a,b,c,j,i,k]
R3a[c,b,a,j,k,i] += X[a,b,c,j,i,k]
R3a[a,b,c,i,j,k] -= X[a,b,c,j,i,k]
R3a[a,c,b,i,j,k] += X[a,b,c,j,i,k]
R3a[b,a,c,i,j,k] += X[a,b,c,j,i,k]
R3a[c,a,b,i,j,k] -= X[a,b,c,j,i,k]
R3a[b,c,a,i,j,k] -= X[a,b,c,j,i,k]
R3a[c,b,a,i,j,k] += X[a,b,c,j,i,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,e,d] * T2a[b,d,i,l] * T2a[c,e,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3a[a,c,b,j,i,k] += X[a,b,c,j,i,k]
R3a[a,b,c,j,k,i] += X[a,b,c,j,i,k]
R3a[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3a[b,a,c,j,i,k] += X[a,b,c,j,i,k]
R3a[c,a,b,j,i,k] -= X[a,b,c,j,i,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,j,i,k]
R3a[c,a,b,j,k,i] += X[a,b,c,j,i,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,j,i,k]
R3a[c,b,a,j,i,k] += X[a,b,c,j,i,k]
R3a[b,c,a,j,k,i] += X[a,b,c,j,i,k]
R3a[c,b,a,j,k,i] -= X[a,b,c,j,i,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,j,i,k]
R3a[b,c,a,i,j,k] += X[a,b,c,j,i,k]
R3a[a,b,c,i,j,k] += X[a,b,c,j,i,k]
R3a[c,b,a,i,j,k] -= X[a,b,c,j,i,k]
R3a[a,c,b,i,j,k] -= X[a,b,c,j,i,k]
R3a[c,a,b,i,j,k] += X[a,b,c,j,i,k]
end
d_vovv = nothing
d_VOVV = load4idx(EC,"d_VOVV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,B=>10*x,C=>10*x) begin
X[a,b,A,i,j,I] := d_VOVV[A,J,B,C] * T2ab[a,B,i,J] * T2ab[b,C,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,C=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_VOVV[A,J,C,B] * T2ab[a,B,i,J] * T2ab[b,C,j,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_VOVV[A,K,D,C] * T2ab[a,B,i,K] * T2b[C,D,I,J]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,J,I]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,J,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,C=>10*x,D=>10*x) begin
X[a,A,B,i,I,J] := d_VOVV[A,K,C,D] * T2b[B,C,I,K] * T2ab[a,D,i,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,C=>10*x,D=>10*x) begin
X[a,A,B,i,I,J] := d_VOVV[A,K,C,D] * T2ab[a,C,i,K] * T2b[B,D,I,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_VOVV[A,K,D,C] * T2b[B,C,I,K] * T2ab[a,D,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_VOVV[A,K,D,C] * T2ab[a,C,i,K] * T2b[B,D,I,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,J=>x,I=>x,K=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,J,I,K] := d_VOVV[A,L,E,D] * T2b[B,C,I,L] * T2b[D,E,J,K]
R3b[A,B,C,J,I,K] += 0.5 * X[A,B,C,J,I,K]
R3b[A,B,C,J,K,I] -= 0.5 * X[A,B,C,J,I,K]
R3b[B,A,C,J,I,K] -= 0.5 * X[A,B,C,J,I,K]
R3b[B,A,C,J,K,I] += 0.5 * X[A,B,C,J,I,K]
R3b[B,C,A,J,I,K] += 0.5 * X[A,B,C,J,I,K]
R3b[B,C,A,J,K,I] -= 0.5 * X[A,B,C,J,I,K]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,J,I,K]
R3b[A,B,C,J,K,I] += 0.5 * X[A,B,C,K,I,J]
R3b[B,A,C,I,J,K] += 0.5 * X[A,B,C,J,I,K]
R3b[B,A,C,J,K,I] -= 0.5 * X[A,B,C,K,I,J]
R3b[B,C,A,I,J,K] -= 0.5 * X[A,B,C,J,I,K]
R3b[B,C,A,J,K,I] += 0.5 * X[A,B,C,K,I,J]
R3b[A,B,C,I,J,K] += 0.5 * X[A,B,C,K,I,J]
R3b[A,B,C,J,I,K] -= 0.5 * X[A,B,C,K,I,J]
R3b[B,A,C,I,J,K] -= 0.5 * X[A,B,C,K,I,J]
R3b[B,A,C,J,I,K] += 0.5 * X[A,B,C,K,I,J]
R3b[B,C,A,I,J,K] += 0.5 * X[A,B,C,K,I,J]
R3b[B,C,A,J,I,K] -= 0.5 * X[A,B,C,K,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,J=>x,I=>x,K=>x,L=>x,D=>10*x,E=>10*x) begin
X[A,B,C,J,I,K] := d_VOVV[A,L,D,E] * T2b[B,D,I,L] * T2b[C,E,J,K]
R3b[A,B,C,J,I,K] += X[A,B,C,J,I,K]
R3b[A,C,B,J,I,K] -= X[A,B,C,J,I,K]
R3b[A,B,C,J,K,I] -= X[A,B,C,J,I,K]
R3b[A,C,B,J,K,I] += X[A,B,C,J,I,K]
R3b[B,A,C,J,I,K] -= X[A,B,C,J,I,K]
R3b[C,A,B,J,I,K] += X[A,B,C,J,I,K]
R3b[B,A,C,J,K,I] += X[A,B,C,J,I,K]
R3b[C,A,B,J,K,I] -= X[A,B,C,J,I,K]
R3b[B,C,A,J,I,K] += X[A,B,C,J,I,K]
R3b[C,B,A,J,I,K] -= X[A,B,C,J,I,K]
R3b[B,C,A,J,K,I] -= X[A,B,C,J,I,K]
R3b[C,B,A,J,K,I] += X[A,B,C,J,I,K]
R3b[A,B,C,I,J,K] -= X[A,B,C,J,I,K]
R3b[A,C,B,I,J,K] += X[A,B,C,J,I,K]
R3b[B,A,C,I,J,K] += X[A,B,C,J,I,K]
R3b[C,A,B,I,J,K] -= X[A,B,C,J,I,K]
R3b[B,C,A,I,J,K] -= X[A,B,C,J,I,K]
R3b[C,B,A,I,J,K] += X[A,B,C,J,I,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,J=>x,I=>x,K=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,J,I,K] := d_VOVV[A,L,E,D] * T2b[B,D,I,L] * T2b[C,E,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,J,I,K]
R3b[A,C,B,J,I,K] += X[A,B,C,J,I,K]
R3b[A,B,C,J,K,I] += X[A,B,C,J,I,K]
R3b[A,C,B,J,K,I] -= X[A,B,C,J,I,K]
R3b[B,A,C,J,I,K] += X[A,B,C,J,I,K]
R3b[C,A,B,J,I,K] -= X[A,B,C,J,I,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,J,I,K]
R3b[C,A,B,J,K,I] += X[A,B,C,J,I,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,J,I,K]
R3b[C,B,A,J,I,K] += X[A,B,C,J,I,K]
R3b[B,C,A,J,K,I] += X[A,B,C,J,I,K]
R3b[C,B,A,J,K,I] -= X[A,B,C,J,I,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,J,I,K]
R3b[B,C,A,I,J,K] += X[A,B,C,J,I,K]
R3b[A,B,C,I,J,K] += X[A,B,C,J,I,K]
R3b[C,B,A,I,J,K] -= X[A,B,C,J,I,K]
R3b[A,C,B,I,J,K] -= X[A,B,C,J,I,K]
R3b[C,A,B,I,J,K] += X[A,B,C,J,I,K]
end
d_VOVV = nothing
d_vOvV = load4idx(EC,"d_vOvV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_vOvV[a,J,c,B] * T2ab[b,A,i,J] * T2ab[c,B,j,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_vOvV[a,J,c,B] * T2ab[c,A,i,J] * T2ab[b,B,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_vOvV[a,J,c,B] * T2ab[b,B,i,J] * T2ab[c,A,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_vOvV[a,J,c,B] * T2b[A,B,I,J] * T2a[b,c,i,j]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_vOvV[a,K,b,C] * T2b[A,B,I,K] * T2ab[b,C,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_vOvV[a,K,b,C] * T2ab[b,A,i,K] * T2b[B,C,I,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_vOvV[a,K,b,C] * T2b[A,C,I,K] * T2ab[b,B,i,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,j,i,k] := d_vOvV[a,I,d,A] * T2ab[b,A,i,I] * T2a[c,d,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3a[a,c,b,j,i,k] += X[a,b,c,j,i,k]
R3a[a,b,c,j,k,i] += X[a,b,c,j,i,k]
R3a[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3a[b,a,c,j,i,k] += X[a,b,c,j,i,k]
R3a[c,a,b,j,i,k] -= X[a,b,c,j,i,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,j,i,k]
R3a[c,a,b,j,k,i] += X[a,b,c,j,i,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,j,i,k]
R3a[c,b,a,j,i,k] += X[a,b,c,j,i,k]
R3a[b,c,a,j,k,i] += X[a,b,c,j,i,k]
R3a[c,b,a,j,k,i] -= X[a,b,c,j,i,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,j,i,k]
R3a[b,c,a,i,j,k] += X[a,b,c,j,i,k]
R3a[a,b,c,i,j,k] += X[a,b,c,j,i,k]
R3a[c,b,a,i,j,k] -= X[a,b,c,j,i,k]
R3a[a,c,b,i,j,k] -= X[a,b,c,j,i,k]
R3a[c,a,b,i,j,k] += X[a,b,c,j,i,k]
end
d_vOvV = nothing
d_vVoV = load4idx(EC,"d_vVoV")
@mtensoropt begin
X[a,b,B,i,j,I] := d_vVoV[a,B,i,A] * T2ab[b,A,j,I]
R3aab[a,b,B,i,j,I] += X[a,b,B,i,j,I]
R3aab[b,a,B,i,j,I] -= X[a,b,B,i,j,I]
R3aab[a,b,B,j,i,I] -= X[a,b,B,i,j,I]
R3aab[b,a,B,j,i,I] += X[a,b,B,i,j,I]
end
@mtensoropt begin
X[a,B,C,i,I,J] := d_vVoV[a,B,i,A] * T2b[C,A,I,J]
R3abb[a,B,C,i,I,J] -= X[a,B,C,i,I,J]
R3abb[a,C,B,i,I,J] += X[a,B,C,i,I,J]
end
d_vVoV = nothing
d_vovo = load4idx(EC,"d_vovo")
@mtensoropt begin
X[b,c,A,j,k,I] := d_vovo[b,i,a,j] * T3aab[c,a,A,k,i,I]
R3aab[b,c,A,j,k,I] -= X[b,c,A,j,k,I]
R3aab[c,b,A,j,k,I] += X[b,c,A,j,k,I]
R3aab[b,c,A,k,j,I] += X[b,c,A,j,k,I]
R3aab[c,b,A,k,j,I] -= X[b,c,A,j,k,I]
end
@mtensoropt R3abb[b,A,B,j,I,J] -= d_vovo[b,i,a,j] * T3abb[a,A,B,i,I,J]
@mtensoropt begin
X[b,c,d,j,k,l] := d_vovo[b,i,a,j] * T3a[c,d,a,k,l,i]
R3a[b,c,d,j,k,l] -= X[b,c,d,j,k,l]
R3a[c,b,d,j,k,l] += X[b,c,d,j,k,l]
R3a[c,d,b,j,k,l] -= X[b,c,d,j,k,l]
R3a[b,c,d,k,j,l] += X[b,c,d,j,k,l]
R3a[c,b,d,k,j,l] -= X[b,c,d,j,k,l]
R3a[c,d,b,k,j,l] += X[b,c,d,j,k,l]
R3a[b,c,d,k,l,j] -= X[b,c,d,j,k,l]
R3a[c,b,d,k,l,j] += X[b,c,d,j,k,l]
R3a[c,d,b,k,l,j] -= X[b,c,d,j,k,l]
end
d_vovo = nothing
d_VOVO = load4idx(EC,"d_VOVO")
@mtensoropt R3aab[a,b,B,i,j,J] -= d_VOVO[B,I,A,J] * T3aab[a,b,A,i,j,I]
@mtensoropt begin
X[a,B,C,i,J,K] := d_VOVO[B,I,A,J] * T3abb[a,C,A,i,K,I]
R3abb[a,B,C,i,J,K] -= X[a,B,C,i,J,K]
R3abb[a,C,B,i,J,K] += X[a,B,C,i,J,K]
R3abb[a,B,C,i,K,J] += X[a,B,C,i,J,K]
R3abb[a,C,B,i,K,J] -= X[a,B,C,i,J,K]
end
@mtensoropt begin
X[B,C,D,J,K,L] := d_VOVO[B,I,A,J] * T3b[C,D,A,K,L,I]
R3b[B,C,D,J,K,L] -= X[B,C,D,J,K,L]
R3b[C,B,D,J,K,L] += X[B,C,D,J,K,L]
R3b[C,D,B,J,K,L] -= X[B,C,D,J,K,L]
R3b[B,C,D,K,J,L] += X[B,C,D,J,K,L]
R3b[C,B,D,K,J,L] -= X[B,C,D,J,K,L]
R3b[C,D,B,K,J,L] += X[B,C,D,J,K,L]
R3b[B,C,D,K,L,J] -= X[B,C,D,J,K,L]
R3b[C,B,D,K,L,J] += X[B,C,D,J,K,L]
R3b[C,D,B,K,L,J] -= X[B,C,D,J,K,L]
end
d_VOVO = nothing
d_vOvO = load4idx(EC,"d_vOvO")
@mtensoropt begin
X[b,c,A,i,j,J] := d_vOvO[b,I,a,J] * T3aab[a,c,A,i,j,I]
R3aab[b,c,A,i,j,J] -= X[b,c,A,i,j,J]
R3aab[c,b,A,i,j,J] += X[b,c,A,i,j,J]
end
@mtensoropt begin
X[b,A,B,i,J,K] := d_vOvO[b,I,a,J] * T3abb[a,B,A,i,K,I]
R3abb[b,A,B,i,J,K] -= X[b,A,B,i,J,K]
R3abb[b,A,B,i,K,J] += X[b,A,B,i,J,K]
end
d_vOvO = nothing
d_voov = load4idx(EC,"d_voov")
@mtensoropt begin
X[b,c,A,j,k,I] := d_voov[b,i,j,a] * T3aab[c,a,A,k,i,I]
R3aab[b,c,A,j,k,I] += X[b,c,A,j,k,I]
R3aab[c,b,A,j,k,I] -= X[b,c,A,j,k,I]
R3aab[b,c,A,k,j,I] -= X[b,c,A,j,k,I]
R3aab[c,b,A,k,j,I] += X[b,c,A,j,k,I]
end
@mtensoropt R3abb[b,A,B,j,I,J] += d_voov[b,i,j,a] * T3abb[a,A,B,i,I,J]
@mtensoropt begin
X[b,c,d,j,k,l] := d_voov[b,i,j,a] * T3a[c,d,a,k,l,i]
R3a[b,c,d,j,k,l] += X[b,c,d,j,k,l]
R3a[c,b,d,j,k,l] -= X[b,c,d,j,k,l]
R3a[c,d,b,j,k,l] += X[b,c,d,j,k,l]
R3a[b,c,d,k,j,l] -= X[b,c,d,j,k,l]
R3a[c,b,d,k,j,l] += X[b,c,d,j,k,l]
R3a[c,d,b,k,j,l] -= X[b,c,d,j,k,l]
R3a[b,c,d,k,l,j] += X[b,c,d,j,k,l]
R3a[c,b,d,k,l,j] -= X[b,c,d,j,k,l]
R3a[c,d,b,k,l,j] += X[b,c,d,j,k,l]
end
d_voov = nothing
d_VOOV = load4idx(EC,"d_VOOV")
@mtensoropt R3aab[a,b,B,i,j,J] += d_VOOV[B,I,J,A] * T3aab[a,b,A,i,j,I]
@mtensoropt begin
X[a,B,C,i,J,K] := d_VOOV[B,I,J,A] * T3abb[a,C,A,i,K,I]
R3abb[a,B,C,i,J,K] += X[a,B,C,i,J,K]
R3abb[a,C,B,i,J,K] -= X[a,B,C,i,J,K]
R3abb[a,B,C,i,K,J] -= X[a,B,C,i,J,K]
R3abb[a,C,B,i,K,J] += X[a,B,C,i,J,K]
end
@mtensoropt begin
X[B,C,D,J,K,L] := d_VOOV[B,I,J,A] * T3b[C,D,A,K,L,I]
R3b[B,C,D,J,K,L] += X[B,C,D,J,K,L]
R3b[C,B,D,J,K,L] -= X[B,C,D,J,K,L]
R3b[C,D,B,J,K,L] += X[B,C,D,J,K,L]
R3b[B,C,D,K,J,L] -= X[B,C,D,J,K,L]
R3b[C,B,D,K,J,L] += X[B,C,D,J,K,L]
R3b[C,D,B,K,J,L] -= X[B,C,D,J,K,L]
R3b[B,C,D,K,L,J] += X[B,C,D,J,K,L]
R3b[C,B,D,K,L,J] -= X[B,C,D,J,K,L]
R3b[C,D,B,K,L,J] += X[B,C,D,J,K,L]
end
d_VOOV = nothing
d_vOoV = load4idx(EC,"d_vOoV")
@mtensoropt begin
X[a,b,B,i,j,J] := d_vOoV[a,I,i,A] * T3abb[b,B,A,j,J,I]
R3aab[a,b,B,i,j,J] += X[a,b,B,i,j,J]
R3aab[b,a,B,i,j,J] -= X[a,b,B,i,j,J]
R3aab[a,b,B,j,i,J] -= X[a,b,B,i,j,J]
R3aab[b,a,B,j,i,J] += X[a,b,B,i,j,J]
end
@mtensoropt R3abb[a,B,C,i,J,K] += d_vOoV[a,I,i,A] * T3b[B,C,A,J,K,I]
@mtensoropt begin
X[a,b,c,i,j,k] := d_vOoV[a,I,i,A] * T3aab[b,c,A,j,k,I]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,i,j,k]
R3a[b,c,a,j,k,i] += X[a,b,c,i,j,k]
end
d_vOoV = nothing
d_vooo = load4idx(EC,"d_vooo")
@mtensoropt begin
X[a,b,A,j,k,I] := d_vooo[a,i,k,j] * T2ab[b,A,i,I]
R3aab[a,b,A,j,k,I] += X[a,b,A,j,k,I]
R3aab[b,a,A,j,k,I] -= X[a,b,A,j,k,I]
R3aab[a,b,A,j,k,I] -= X[a,b,A,k,j,I]
R3aab[b,a,A,j,k,I] += X[a,b,A,k,j,I]
end
@mtensoropt begin
X[b,a,c,j,k,l] := d_vooo[a,i,k,j] * T2a[b,c,l,i]
R3a[b,a,c,j,k,l] += X[b,a,c,j,k,l]
R3a[b,c,a,j,k,l] -= X[b,a,c,j,k,l]
R3a[b,a,c,j,l,k] -= X[b,a,c,j,k,l]
R3a[b,c,a,j,l,k] += X[b,a,c,j,k,l]
R3a[a,b,c,j,k,l] -= X[b,a,c,j,k,l]
R3a[a,b,c,j,l,k] += X[b,a,c,j,k,l]
R3a[b,a,c,j,k,l] -= X[b,a,c,k,j,l]
R3a[b,c,a,j,k,l] += X[b,a,c,k,j,l]
R3a[b,a,c,l,j,k] += X[b,a,c,j,k,l]
R3a[b,c,a,l,j,k] -= X[b,a,c,j,k,l]
R3a[a,b,c,j,k,l] += X[b,a,c,k,j,l]
R3a[a,b,c,l,j,k] -= X[b,a,c,j,k,l]
R3a[b,a,c,j,l,k] += X[b,a,c,k,j,l]
R3a[b,c,a,j,l,k] -= X[b,a,c,k,j,l]
R3a[b,a,c,l,j,k] -= X[b,a,c,k,j,l]
R3a[b,c,a,l,j,k] += X[b,a,c,k,j,l]
R3a[a,b,c,j,l,k] -= X[b,a,c,k,j,l]
R3a[a,b,c,l,j,k] += X[b,a,c,k,j,l]
end
d_vooo = nothing
d_VOOO = load4idx(EC,"d_VOOO")
@mtensoropt begin
X[a,A,B,i,J,K] := d_VOOO[A,I,K,J] * T2ab[a,B,i,I]
R3abb[a,A,B,i,J,K] += X[a,A,B,i,J,K]
R3abb[a,B,A,i,J,K] -= X[a,A,B,i,J,K]
R3abb[a,A,B,i,J,K] -= X[a,A,B,i,K,J]
R3abb[a,B,A,i,J,K] += X[a,A,B,i,K,J]
end
@mtensoropt begin
X[B,A,C,J,K,L] := d_VOOO[A,I,K,J] * T2b[B,C,L,I]
R3b[B,A,C,J,K,L] += X[B,A,C,J,K,L]
R3b[B,C,A,J,K,L] -= X[B,A,C,J,K,L]
R3b[B,A,C,J,L,K] -= X[B,A,C,J,K,L]
R3b[B,C,A,J,L,K] += X[B,A,C,J,K,L]
R3b[A,B,C,J,K,L] -= X[B,A,C,J,K,L]
R3b[A,B,C,J,L,K] += X[B,A,C,J,K,L]
R3b[B,A,C,J,K,L] -= X[B,A,C,K,J,L]
R3b[B,C,A,J,K,L] += X[B,A,C,K,J,L]
R3b[B,A,C,L,J,K] += X[B,A,C,J,K,L]
R3b[B,C,A,L,J,K] -= X[B,A,C,J,K,L]
R3b[A,B,C,J,K,L] += X[B,A,C,K,J,L]
R3b[A,B,C,L,J,K] -= X[B,A,C,J,K,L]
R3b[B,A,C,J,L,K] += X[B,A,C,K,J,L]
R3b[B,C,A,J,L,K] -= X[B,A,C,K,J,L]
R3b[B,A,C,L,J,K] -= X[B,A,C,K,J,L]
R3b[B,C,A,L,J,K] += X[B,A,C,K,J,L]
R3b[A,B,C,J,L,K] -= X[B,A,C,K,J,L]
R3b[A,B,C,L,J,K] += X[B,A,C,K,J,L]
end
d_VOOO = nothing
d_vOoO = load4idx(EC,"d_vOoO")
@mtensoropt begin
X[a,b,A,i,j,J] := d_vOoO[a,I,i,J] * T2ab[b,A,j,I]
R3aab[a,b,A,i,j,J] -= X[a,b,A,i,j,J]
R3aab[b,a,A,i,j,J] += X[a,b,A,i,j,J]
R3aab[a,b,A,j,i,J] += X[a,b,A,i,j,J]
R3aab[b,a,A,j,i,J] -= X[a,b,A,i,j,J]
end
@mtensoropt begin
X[a,A,B,i,J,K] := d_vOoO[a,I,i,J] * T2b[A,B,K,I]
R3abb[a,A,B,i,J,K] += X[a,A,B,i,J,K]
R3abb[a,A,B,i,K,J] -= X[a,A,B,i,J,K]
end
d_vOoO = nothing
d_oooo = load4idx(EC,"d_oooo")
@mtensoropt begin
X[a,b,A,k,l,I] := d_oooo[j,i,l,k] * T3aab[a,b,A,i,j,I]
R3aab[a,b,A,k,l,I] += 0.5 * X[a,b,A,k,l,I]
R3aab[a,b,A,k,l,I] -= 0.5 * X[b,a,A,k,l,I]
end
@mtensoropt begin
X[a,b,c,k,l,m] := d_oooo[j,i,l,k] * T3a[a,b,c,m,i,j]
R3a[a,b,c,k,l,m] += 0.5 * X[a,b,c,k,l,m]
R3a[a,b,c,k,l,m] -= 0.5 * X[a,c,b,k,l,m]
R3a[a,b,c,k,m,l] -= 0.5 * X[a,b,c,k,l,m]
R3a[a,b,c,k,m,l] += 0.5 * X[a,c,b,k,l,m]
R3a[a,b,c,m,k,l] += 0.5 * X[a,b,c,k,l,m]
R3a[a,b,c,m,k,l] -= 0.5 * X[a,c,b,k,l,m]
end
d_oooo = nothing
d_OOOO = load4idx(EC,"d_OOOO")
@mtensoropt begin
X[a,A,B,i,K,L] := d_OOOO[J,I,L,K] * T3abb[a,A,B,i,I,J]
R3abb[a,A,B,i,K,L] += 0.5 * X[a,A,B,i,K,L]
R3abb[a,A,B,i,K,L] -= 0.5 * X[a,B,A,i,K,L]
end
@mtensoropt begin
X[A,B,C,K,L,M] := d_OOOO[J,I,L,K] * T3b[A,B,C,M,I,J]
R3b[A,B,C,K,L,M] += 0.5 * X[A,B,C,K,L,M]
R3b[A,B,C,K,L,M] -= 0.5 * X[A,C,B,K,L,M]
R3b[A,B,C,K,M,L] -= 0.5 * X[A,B,C,K,L,M]
R3b[A,B,C,K,M,L] += 0.5 * X[A,C,B,K,L,M]
R3b[A,B,C,M,K,L] += 0.5 * X[A,B,C,K,L,M]
R3b[A,B,C,M,K,L] -= 0.5 * X[A,C,B,K,L,M]
end
d_OOOO = nothing
d_oOoO = load4idx(EC,"d_oOoO")
@mtensoropt begin
X[a,b,A,j,k,J] := d_oOoO[i,I,j,J] * T3aab[a,b,A,k,i,I]
R3aab[a,b,A,j,k,J] -= 0.5 * X[a,b,A,j,k,J]
R3aab[a,b,A,j,k,J] += 0.5 * X[b,a,A,j,k,J]
R3aab[a,b,A,k,j,J] += 0.5 * X[a,b,A,j,k,J]
R3aab[a,b,A,k,j,J] -= 0.5 * X[b,a,A,j,k,J]
end
@mtensoropt begin
X[a,A,B,j,J,K] := d_oOoO[i,I,j,J] * T3abb[a,A,B,i,K,I]
R3abb[a,A,B,j,J,K] -= 0.5 * X[a,A,B,j,J,K]
R3abb[a,A,B,j,J,K] += 0.5 * X[a,B,A,j,J,K]
R3abb[a,A,B,j,K,J] += 0.5 * X[a,A,B,j,J,K]
R3abb[a,A,B,j,K,J] -= 0.5 * X[a,B,A,j,J,K]
end
d_oOoO = nothing
d_oVoO = load4idx(EC,"d_oVoO")
@mtensoropt begin
X[a,b,A,j,k,I] := d_oVoO[i,A,j,I] * T2a[a,b,k,i]
R3aab[a,b,A,j,k,I] += X[a,b,A,j,k,I]
R3aab[a,b,A,k,j,I] -= X[a,b,A,j,k,I]
end
@mtensoropt begin
X[a,A,B,j,I,J] := d_oVoO[i,A,j,I] * T2ab[a,B,i,J]
R3abb[a,A,B,j,I,J] -= X[a,A,B,j,I,J]
R3abb[a,B,A,j,I,J] += X[a,A,B,j,I,J]
R3abb[a,A,B,j,J,I] += X[a,A,B,j,I,J]
R3abb[a,B,A,j,J,I] -= X[a,A,B,j,I,J]
end
d_oVoO = nothing
d_oovo = load4idx(EC,"d_oovo")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oovo[l,k,c,i] * T2a[a,b,k,l] * T2ab[c,A,j,I]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,i,j,I] -= 0.5 * X[b,a,A,i,j,I]
R3aab[a,b,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[b,a,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oovo[l,k,c,i] * T2ab[c,A,k,I] * T2a[a,b,j,l]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (b=>10*x,a=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x) begin
X[b,a,A,i,j,I] := d_oovo[l,k,c,i] * T2a[a,c,j,k] * T2ab[b,A,l,I]
R3aab[b,a,A,i,j,I] += X[b,a,A,i,j,I]
R3aab[a,b,A,i,j,I] -= X[b,a,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[b,a,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[b,a,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oovo[l,k,c,i] * T2ab[a,A,k,I] * T2a[b,c,j,l]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oovo[l,k,c,i] * T2a[a,b,j,k] * T2ab[c,A,l,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oovo[k,j,b,i] * T2ab[b,A,j,I] * T2ab[a,B,k,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oovo[k,j,b,i] * T2ab[a,A,j,I] * T2ab[b,B,k,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (b=>10*x,c=>10*x,a=>10*x,k=>x,i=>x,j=>x,m=>x,l=>x,d=>10*x) begin
X[b,c,a,k,i,j] := d_oovo[m,l,d,i] * T2a[a,d,j,l] * T2a[b,c,k,m]
R3a[b,c,a,k,i,j] += X[b,c,a,k,i,j]
R3a[b,a,c,k,i,j] -= X[b,c,a,k,i,j]
R3a[b,c,a,k,j,i] -= X[b,c,a,k,i,j]
R3a[b,a,c,k,j,i] += X[b,c,a,k,i,j]
R3a[a,b,c,k,i,j] += X[b,c,a,k,i,j]
R3a[a,b,c,k,j,i] -= X[b,c,a,k,i,j]
end
@mtensoropt (b=>10*x,a=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[b,a,c,i,j,k] := d_oovo[m,l,d,i] * T2a[a,d,j,l] * T2a[b,c,k,m]
R3a[b,a,c,i,j,k] -= X[b,a,c,i,j,k]
R3a[b,c,a,i,j,k] += X[b,a,c,i,j,k]
R3a[b,a,c,i,k,j] += X[b,a,c,i,j,k]
R3a[b,c,a,i,k,j] -= X[b,a,c,i,j,k]
R3a[a,b,c,i,j,k] += X[b,a,c,i,j,k]
R3a[a,b,c,i,k,j] -= X[b,a,c,i,j,k]
R3a[b,a,c,j,i,k] += X[b,a,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[b,a,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[b,a,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[b,a,c,i,j,k]
R3a[b,c,a,j,k,i] += X[b,a,c,i,j,k]
R3a[a,b,c,j,k,i] += X[b,a,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2a[a,b,l,m] * T2a[c,d,j,k]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,i,j,k] += 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,i,j,k] += 0.5 * X[b,a,c,i,j,k]
R3a[a,b,c,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,j,i,k] -= 0.5 * X[b,a,c,i,j,k]
R3a[a,b,c,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= 0.5 * X[b,a,c,i,j,k]
R3a[c,a,b,j,k,i] += 0.5 * X[b,a,c,i,j,k]
end
@mtensoropt (c=>10*x,a=>10*x,b=>10*x,k=>x,i=>x,j=>x,m=>x,l=>x,d=>10*x) begin
X[c,a,b,k,i,j] := d_oovo[m,l,d,i] * T2a[a,b,j,l] * T2a[c,d,k,m]
R3a[c,a,b,k,i,j] += X[c,a,b,k,i,j]
R3a[c,a,b,k,j,i] -= X[c,a,b,k,i,j]
R3a[a,c,b,k,i,j] -= X[c,a,b,k,i,j]
R3a[a,c,b,k,j,i] += X[c,a,b,k,i,j]
R3a[a,b,c,k,i,j] += X[c,a,b,k,i,j]
R3a[a,b,c,k,j,i] -= X[c,a,b,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2a[a,b,j,l] * T2a[c,d,k,m]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,i,k,j] -= X[a,b,c,i,j,k]
R3a[a,c,b,i,k,j] += X[a,b,c,i,j,k]
R3a[c,a,b,i,j,k] += X[a,b,c,i,j,k]
R3a[c,a,b,i,k,j] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += X[a,b,c,i,j,k]
R3a[c,a,b,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= X[a,b,c,i,j,k]
R3a[c,a,b,j,k,i] += X[a,b,c,i,j,k]
end
d_oovo = nothing
d_OOVO = load4idx(EC,"d_OOVO")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_OOVO[K,J,B,I] * T2ab[a,B,i,J] * T2ab[b,A,j,K]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_OOVO[K,J,B,I] * T2ab[a,A,i,J] * T2ab[b,B,j,K]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_OOVO[L,K,C,I] * T2b[A,B,K,L] * T2ab[a,C,i,J]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,B,A,i,I,J]
R3abb[a,A,B,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,B,A,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_OOVO[L,K,C,I] * T2ab[a,C,i,K] * T2b[A,B,J,L]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,B=>10*x,A=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x) begin
X[a,B,A,i,I,J] := d_OOVO[L,K,C,I] * T2b[A,C,J,K] * T2ab[a,B,i,L]
R3abb[a,B,A,i,I,J] += X[a,B,A,i,I,J]
R3abb[a,A,B,i,I,J] -= X[a,B,A,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,B,A,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,B,A,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_OOVO[L,K,C,I] * T2ab[a,A,i,K] * T2b[B,C,J,L]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_OOVO[L,K,C,I] * T2b[A,B,J,K] * T2ab[a,C,i,L]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (B=>10*x,C=>10*x,A=>10*x,K=>x,I=>x,J=>x,M=>x,L=>x,D=>10*x) begin
X[B,C,A,K,I,J] := d_OOVO[M,L,D,I] * T2b[A,D,J,L] * T2b[B,C,K,M]
R3b[B,C,A,K,I,J] += X[B,C,A,K,I,J]
R3b[B,A,C,K,I,J] -= X[B,C,A,K,I,J]
R3b[B,C,A,K,J,I] -= X[B,C,A,K,I,J]
R3b[B,A,C,K,J,I] += X[B,C,A,K,I,J]
R3b[A,B,C,K,I,J] += X[B,C,A,K,I,J]
R3b[A,B,C,K,J,I] -= X[B,C,A,K,I,J]
end
@mtensoropt (B=>10*x,A=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,D=>10*x) begin
X[B,A,C,I,J,K] := d_OOVO[M,L,D,I] * T2b[A,D,J,L] * T2b[B,C,K,M]
R3b[B,A,C,I,J,K] -= X[B,A,C,I,J,K]
R3b[B,C,A,I,J,K] += X[B,A,C,I,J,K]
R3b[B,A,C,I,K,J] += X[B,A,C,I,J,K]
R3b[B,C,A,I,K,J] -= X[B,A,C,I,J,K]
R3b[A,B,C,I,J,K] += X[B,A,C,I,J,K]
R3b[A,B,C,I,K,J] -= X[B,A,C,I,J,K]
R3b[B,A,C,J,I,K] += X[B,A,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[B,A,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[B,A,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[B,A,C,I,J,K]
R3b[B,C,A,J,K,I] += X[B,A,C,I,J,K]
R3b[A,B,C,J,K,I] += X[B,A,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,D=>10*x) begin
X[A,B,C,I,J,K] := d_OOVO[M,L,D,I] * T2b[A,B,L,M] * T2b[C,D,J,K]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,I,J,K] += 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,I,J,K] += 0.5 * X[B,A,C,I,J,K]
R3b[A,B,C,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,J,I,K] -= 0.5 * X[B,A,C,I,J,K]
R3b[A,B,C,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= 0.5 * X[B,A,C,I,J,K]
R3b[C,A,B,J,K,I] += 0.5 * X[B,A,C,I,J,K]
end
@mtensoropt (C=>10*x,A=>10*x,B=>10*x,K=>x,I=>x,J=>x,M=>x,L=>x,D=>10*x) begin
X[C,A,B,K,I,J] := d_OOVO[M,L,D,I] * T2b[A,B,J,L] * T2b[C,D,K,M]
R3b[C,A,B,K,I,J] += X[C,A,B,K,I,J]
R3b[C,A,B,K,J,I] -= X[C,A,B,K,I,J]
R3b[A,C,B,K,I,J] -= X[C,A,B,K,I,J]
R3b[A,C,B,K,J,I] += X[C,A,B,K,I,J]
R3b[A,B,C,K,I,J] += X[C,A,B,K,I,J]
R3b[A,B,C,K,J,I] -= X[C,A,B,K,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,D=>10*x) begin
X[A,B,C,I,J,K] := d_OOVO[M,L,D,I] * T2b[A,B,J,L] * T2b[C,D,K,M]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,I,K,J] -= X[A,B,C,I,J,K]
R3b[A,C,B,I,K,J] += X[A,B,C,I,J,K]
R3b[C,A,B,I,J,K] += X[A,B,C,I,J,K]
R3b[C,A,B,I,K,J] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += X[A,B,C,I,J,K]
R3b[C,A,B,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= X[A,B,C,I,J,K]
R3b[C,A,B,J,K,I] += X[A,B,C,I,J,K]
end
d_OOVO = nothing
d_oOvO = load4idx(EC,"d_oOvO")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oOvO[k,J,c,I] * T2ab[c,A,i,J] * T2a[a,b,j,k]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oOvO[k,J,c,I] * T2ab[a,A,k,J] * T2a[b,c,i,j]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x) begin
X[a,b,A,i,j,I] := d_oOvO[k,J,c,I] * T2ab[a,A,i,J] * T2a[b,c,j,k]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oOvO[j,K,b,I] * T2ab[b,A,i,K] * T2ab[a,B,j,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oOvO[j,K,b,I] * T2ab[a,A,j,K] * T2ab[b,B,i,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oOvO[j,K,b,I] * T2ab[a,A,i,K] * T2ab[b,B,j,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x) begin
X[a,A,B,i,I,J] := d_oOvO[j,K,b,I] * T2b[A,B,J,K] * T2a[a,b,i,j]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (C=>10*x,A=>10*x,B=>10*x,K=>x,I=>x,J=>x,i=>x,L=>x,a=>10*x) begin
X[C,A,B,K,I,J] := d_oOvO[i,L,a,I] * T2b[A,B,J,L] * T2ab[a,C,i,K]
R3b[C,A,B,K,I,J] += X[C,A,B,K,I,J]
R3b[C,A,B,K,J,I] -= X[C,A,B,K,I,J]
R3b[A,C,B,K,I,J] -= X[C,A,B,K,I,J]
R3b[A,C,B,K,J,I] += X[C,A,B,K,I,J]
R3b[A,B,C,K,I,J] += X[C,A,B,K,I,J]
R3b[A,B,C,K,J,I] -= X[C,A,B,K,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x) begin
X[A,B,C,I,J,K] := d_oOvO[i,L,a,I] * T2b[A,B,J,L] * T2ab[a,C,i,K]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,I,K,J] -= X[A,B,C,I,J,K]
R3b[A,C,B,I,K,J] += X[A,B,C,I,J,K]
R3b[C,A,B,I,J,K] += X[A,B,C,I,J,K]
R3b[C,A,B,I,K,J] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += X[A,B,C,I,J,K]
R3b[C,A,B,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= X[A,B,C,I,J,K]
R3b[C,A,B,J,K,I] += X[A,B,C,I,J,K]
end
d_oOvO = nothing
d_oOoV = load4idx(EC,"d_oOoV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oOoV[k,J,i,B] * T2ab[a,B,k,I] * T2ab[b,A,j,J]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oOoV[k,J,i,B] * T2ab[a,A,k,J] * T2ab[b,B,j,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oOoV[k,J,i,B] * T2ab[a,A,k,I] * T2ab[b,B,j,J]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oOoV[k,J,i,B] * T2a[a,b,j,k] * T2b[A,B,I,J]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oOoV[j,K,i,C] * T2ab[a,C,j,I] * T2b[A,B,J,K]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oOoV[j,K,i,C] * T2ab[a,A,j,K] * T2b[B,C,I,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oOoV[j,K,i,C] * T2ab[a,A,j,I] * T2b[B,C,J,K]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (c=>10*x,a=>10*x,b=>10*x,k=>x,i=>x,j=>x,l=>x,I=>x,A=>10*x) begin
X[c,a,b,k,i,j] := d_oOoV[l,I,i,A] * T2a[a,b,j,l] * T2ab[c,A,k,I]
R3a[c,a,b,k,i,j] += X[c,a,b,k,i,j]
R3a[c,a,b,k,j,i] -= X[c,a,b,k,i,j]
R3a[a,c,b,k,i,j] -= X[c,a,b,k,i,j]
R3a[a,c,b,k,j,i] += X[c,a,b,k,i,j]
R3a[a,b,c,k,i,j] += X[c,a,b,k,i,j]
R3a[a,b,c,k,j,i] -= X[c,a,b,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,A=>10*x) begin
X[a,b,c,i,j,k] := d_oOoV[l,I,i,A] * T2a[a,b,j,l] * T2ab[c,A,k,I]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,i,k,j] -= X[a,b,c,i,j,k]
R3a[a,c,b,i,k,j] += X[a,b,c,i,j,k]
R3a[c,a,b,i,j,k] += X[a,b,c,i,j,k]
R3a[c,a,b,i,k,j] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += X[a,b,c,i,j,k]
R3a[c,a,b,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= X[a,b,c,i,j,k]
R3a[c,a,b,j,k,i] += X[a,b,c,i,j,k]
end
d_oOoV = nothing
d_oVvO = load4idx(EC,"d_oVvO")
@mtensoropt R3aab[b,c,A,j,k,I] += d_oVvO[i,A,a,I] * T3a[b,c,a,j,k,i]
@mtensoropt begin
X[b,A,B,j,I,J] := d_oVvO[i,A,a,I] * T3aab[b,a,B,j,i,J]
R3abb[b,A,B,j,I,J] += X[b,A,B,j,I,J]
R3abb[b,B,A,j,I,J] -= X[b,A,B,j,I,J]
R3abb[b,A,B,j,J,I] -= X[b,A,B,j,I,J]
R3abb[b,B,A,j,J,I] += X[b,A,B,j,I,J]
end
@mtensoropt begin
X[A,B,C,I,J,K] := d_oVvO[i,A,a,I] * T3abb[a,B,C,i,J,K]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[B,A,C,J,I,K] += X[A,B,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,I,J,K]
R3b[B,C,A,J,K,I] += X[A,B,C,I,J,K]
end
d_oVvO = nothing
d_oVoV = load4idx(EC,"d_oVoV")
@mtensoropt begin
X[a,b,B,j,k,I] := d_oVoV[i,B,j,A] * T3aab[b,a,A,k,i,I]
R3aab[a,b,B,j,k,I] -= X[a,b,B,j,k,I]
R3aab[a,b,B,k,j,I] += X[a,b,B,j,k,I]
end
@mtensoropt begin
X[a,B,C,j,I,J] := d_oVoV[i,B,j,A] * T3abb[a,A,C,i,I,J]
R3abb[a,B,C,j,I,J] -= X[a,B,C,j,I,J]
R3abb[a,C,B,j,I,J] += X[a,B,C,j,I,J]
end
d_oVoV = nothing
oovv = ints2(EC,"oovv")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,c=>10*x,d=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,c,d] * T2a[c,d,i,k] * T3aab[a,b,A,j,l,I]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[c,a,k,l] * T3aab[b,d,A,i,j,I]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[c,d,i,j] * T3aab[a,b,A,k,l,I]
R3aab[a,b,A,i,j,I] -= 0.25 * X[a,b,A,i,j,I]
R3aab[a,b,A,i,j,I] -= 0.25 * X[b,a,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[a,b,k,l] * T3aab[c,d,A,i,j,I]
R3aab[a,b,A,i,j,I] -= 0.25 * X[a,b,A,i,j,I]
R3aab[a,b,A,i,j,I] -= 0.25 * X[b,a,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[a,c,i,j] * T3aab[b,d,A,k,l,I]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[a,b,i,k] * T3aab[c,d,A,j,l,I]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2ab[c,A,i,I] * T3a[a,b,d,j,k,l]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2ab[a,A,k,I] * T3a[b,c,d,i,j,l]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) begin
X[a,b,A,i,j,I] := oovv[l,k,d,c] * T2a[a,c,i,k] * T3aab[b,d,A,j,l,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,l=>x,k=>x,d=>10*x,c=>10*x) R3aab[a,b,A,i,j,I] += oovv[l,k,d,c] * T2ab[c,A,k,I] * T3a[a,b,d,i,j,l]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,b=>10*x,c=>10*x) R3abb[a,A,B,i,I,J] += 0.5 * oovv[k,j,b,c] * T2a[b,c,i,j] * T3abb[a,A,B,k,J,I]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,c=>10*x,b=>10*x) R3abb[a,A,B,i,I,J] += 0.5 * oovv[k,j,c,b] * T2a[b,a,j,k] * T3abb[c,B,A,i,I,J]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,c=>10*x,b=>10*x) begin
X[a,A,B,i,I,J] := oovv[k,j,c,b] * T2ab[a,A,j,I] * T3aab[b,c,B,i,k,J]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,c=>10*x,b=>10*x) begin
X[a,A,B,i,I,J] := oovv[k,j,c,b] * T2ab[b,A,i,I] * T3aab[a,c,B,j,k,J]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,c=>10*x,b=>10*x) begin
X[a,A,B,i,I,J] := oovv[k,j,c,b] * T2ab[b,A,j,I] * T3aab[a,c,B,i,k,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,k=>x,j=>x,c=>10*x,b=>10*x) R3abb[a,A,B,i,I,J] += oovv[k,j,c,b] * T2a[a,b,i,j] * T3abb[c,A,B,k,I,J]
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,j=>x,i=>x,b=>10*x,a=>10*x) begin
X[A,B,C,I,J,K] := oovv[j,i,b,a] * T2ab[a,A,i,I] * T3abb[b,B,C,j,J,K]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[B,A,C,J,I,K] += X[A,B,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,I,J,K]
R3b[B,C,A,J,K,I] += X[A,B,C,I,J,K]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,d,e] * T2a[d,e,i,l] * T3a[a,b,c,j,k,m]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[d,a,l,m] * T3a[b,c,e,i,j,k]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[d,e,i,j] * T3a[a,b,c,k,l,m]
R3a[a,b,c,i,j,k] -= 0.25 * X[a,b,c,i,j,k]
R3a[a,b,c,i,j,k] -= 0.25 * X[a,c,b,i,j,k]
R3a[a,b,c,i,k,j] += 0.25 * X[a,b,c,i,j,k]
R3a[a,b,c,i,k,j] += 0.25 * X[a,c,b,i,j,k]
R3a[a,b,c,k,i,j] -= 0.25 * X[a,b,c,i,j,k]
R3a[a,b,c,k,i,j] -= 0.25 * X[a,c,b,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[a,b,l,m] * T3a[c,d,e,i,j,k]
R3a[a,b,c,i,j,k] -= 0.25 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] += 0.25 * X[a,b,c,i,j,k]
R3a[a,b,c,i,j,k] -= 0.25 * X[b,a,c,i,j,k]
R3a[c,a,b,i,j,k] -= 0.25 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] += 0.25 * X[b,a,c,i,j,k]
R3a[c,a,b,i,j,k] -= 0.25 * X[b,a,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[a,d,i,j] * T3a[b,c,e,k,l,m]
R3a[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,i,k,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,k,j] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,k,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,k,i,j] += 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,k,i,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,k,i,j] += 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[a,b,i,l] * T3a[c,d,e,j,k,m]
R3a[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,j,i,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,j,k,i] += 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2a[a,d,i,l] * T3a[b,c,e,j,k,m]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,i,j,k]
R3a[b,c,a,j,k,i] += X[a,b,c,i,j,k]
end
oovv = nothing
OOVV = ints2(EC,"OOVV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,B=>10*x,C=>10*x) R3aab[a,b,A,i,j,I] += 0.5 * OOVV[K,J,B,C] * T2b[B,C,I,J] * T3aab[a,b,A,j,i,K]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += 0.5 * OOVV[K,J,C,B] * T2b[B,A,J,K] * T3aab[b,a,C,i,j,I]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := OOVV[K,J,C,B] * T2ab[a,A,i,J] * T3abb[b,B,C,j,I,K]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := OOVV[K,J,C,B] * T2ab[a,B,i,I] * T3abb[b,A,C,j,J,K]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := OOVV[K,J,C,B] * T2ab[a,B,i,J] * T3abb[b,A,C,j,I,K]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,K=>x,J=>x,C=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += OOVV[K,J,C,B] * T2b[A,B,I,J] * T3aab[a,b,C,i,j,K]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,C=>10*x,D=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,C,D] * T2b[C,D,I,K] * T3abb[a,A,B,i,J,L]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[C,A,K,L] * T3abb[a,B,D,i,I,J]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[C,D,I,J] * T3abb[a,A,B,i,K,L]
R3abb[a,A,B,i,I,J] -= 0.25 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,I,J] -= 0.25 * X[a,B,A,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[A,B,K,L] * T3abb[a,C,D,i,I,J]
R3abb[a,A,B,i,I,J] -= 0.25 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,I,J] -= 0.25 * X[a,B,A,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[A,C,I,J] * T3abb[a,B,D,i,K,L]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[A,B,I,K] * T3abb[a,C,D,i,J,L]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2ab[a,C,i,I] * T3b[A,B,D,J,K,L]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2ab[a,A,i,K] * T3b[B,C,D,I,J,L]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := OOVV[L,K,D,C] * T2b[A,C,I,K] * T3abb[a,B,D,i,J,L]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,L=>x,K=>x,D=>10*x,C=>10*x) R3abb[a,A,B,i,I,J] += OOVV[L,K,D,C] * T2ab[a,C,i,K] * T3b[A,B,D,I,J,L]
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,D=>10*x,E=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,D,E] * T2b[D,E,I,L] * T3b[A,B,C,J,K,M]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[D,A,L,M] * T3b[B,C,E,I,J,K]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[D,E,I,J] * T3b[A,B,C,K,L,M]
R3b[A,B,C,I,J,K] -= 0.25 * X[A,B,C,I,J,K]
R3b[A,B,C,I,J,K] -= 0.25 * X[A,C,B,I,J,K]
R3b[A,B,C,I,K,J] += 0.25 * X[A,B,C,I,J,K]
R3b[A,B,C,I,K,J] += 0.25 * X[A,C,B,I,J,K]
R3b[A,B,C,K,I,J] -= 0.25 * X[A,B,C,I,J,K]
R3b[A,B,C,K,I,J] -= 0.25 * X[A,C,B,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[A,B,L,M] * T3b[C,D,E,I,J,K]
R3b[A,B,C,I,J,K] -= 0.25 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] += 0.25 * X[A,B,C,I,J,K]
R3b[A,B,C,I,J,K] -= 0.25 * X[B,A,C,I,J,K]
R3b[C,A,B,I,J,K] -= 0.25 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] += 0.25 * X[B,A,C,I,J,K]
R3b[C,A,B,I,J,K] -= 0.25 * X[B,A,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[A,D,I,J] * T3b[B,C,E,K,L,M]
R3b[A,B,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,I,K,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,K,J] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,K,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,K,I,J] += 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,K,I,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,K,I,J] += 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[A,B,I,L] * T3b[C,D,E,J,K,M]
R3b[A,B,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,J,I,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,J,K,I] += 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,M=>x,L=>x,E=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := OOVV[M,L,E,D] * T2b[A,D,I,L] * T3b[B,C,E,J,K,M]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[B,A,C,J,I,K] += X[A,B,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,I,J,K]
R3b[B,C,A,J,K,I] += X[A,B,C,I,J,K]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,J=>x,I=>x,B=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := OOVV[J,I,B,A] * T2ab[a,A,i,I] * T3aab[b,c,B,j,k,J]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,i,j,k]
R3a[b,c,a,j,k,i] += X[a,b,c,i,j,k]
end
OOVV = nothing
oOvV = ints2(EC,"oOvV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[c,B,i,J] * T3aab[a,b,A,j,k,I]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[a,B,k,J] * T3aab[b,c,A,i,j,I]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += 0.5 * oOvV[k,J,c,B] * T2ab[c,B,k,I] * T3aab[a,b,A,j,i,J]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += 0.5 * oOvV[k,J,c,B] * T2ab[c,A,k,J] * T3aab[b,a,B,i,j,I]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2a[a,c,i,j] * T3abb[b,A,B,k,I,J]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2a[a,b,i,k] * T3abb[c,A,B,j,I,J]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[a,A,i,J] * T3aab[b,c,B,j,k,I]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[a,B,i,I] * T3aab[b,c,A,j,k,J]
R3aab[a,b,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[c,A,i,I] * T3aab[a,b,B,j,k,J]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[a,A,k,I] * T3aab[b,c,B,i,j,J]
R3aab[a,b,A,i,j,I] += 0.5 * X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= 0.5 * X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2ab[a,B,i,J] * T3aab[b,c,A,j,k,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := oOvV[k,J,c,B] * T2a[a,c,i,k] * T3abb[b,A,B,j,I,J]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += oOvV[k,J,c,B] * T2b[A,B,I,J] * T3a[a,b,c,i,j,k]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,J=>x,c=>10*x,B=>10*x) R3aab[a,b,A,i,j,I] += oOvV[k,J,c,B] * T2ab[c,A,k,I] * T3aab[a,b,B,i,j,J]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[b,C,j,I] * T3abb[a,A,B,i,J,K]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[b,A,j,K] * T3abb[a,B,C,i,I,J]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) R3abb[a,A,B,i,I,J] += 0.5 * oOvV[j,K,b,C] * T2ab[b,C,i,K] * T3abb[a,A,B,j,J,I]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) R3abb[a,A,B,i,I,J] += 0.5 * oOvV[j,K,b,C] * T2ab[a,C,j,K] * T3abb[b,B,A,i,I,J]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2b[A,C,I,J] * T3aab[a,b,B,i,j,K]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2b[A,B,I,K] * T3aab[a,b,C,i,j,J]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[a,A,j,I] * T3abb[b,B,C,i,J,K]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[b,A,i,I] * T3abb[a,B,C,j,J,K]
R3abb[a,A,B,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[a,C,i,I] * T3abb[b,A,B,j,J,K]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[a,A,i,K] * T3abb[b,B,C,j,I,J]
R3abb[a,A,B,i,I,J] += 0.5 * X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= 0.5 * X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2b[A,C,I,K] * T3aab[a,b,B,i,j,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := oOvV[j,K,b,C] * T2ab[b,A,j,I] * T3abb[a,B,C,i,J,K]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) R3abb[a,A,B,i,I,J] += oOvV[j,K,b,C] * T2ab[a,C,i,K] * T3abb[b,A,B,j,I,J]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,K=>x,b=>10*x,C=>10*x) R3abb[a,A,B,i,I,J] += oOvV[j,K,b,C] * T2a[a,b,i,j] * T3b[A,B,C,I,J,K]
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2ab[a,D,i,I] * T3b[A,B,C,J,K,L]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2ab[a,A,i,L] * T3b[B,C,D,I,J,K]
R3b[A,B,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2b[A,D,I,J] * T3abb[a,B,C,i,K,L]
R3b[A,B,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,I,K,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,I,K,J] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,I,K,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,K,I,J] += 0.5 * X[A,B,C,I,J,K]
R3b[B,A,C,K,I,J] -= 0.5 * X[A,B,C,I,J,K]
R3b[B,C,A,K,I,J] += 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2b[A,B,I,L] * T3abb[a,C,D,i,J,K]
R3b[A,B,C,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,I,J,K] += 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,J,I,K] -= 0.5 * X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += 0.5 * X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= 0.5 * X[A,B,C,I,J,K]
R3b[C,A,B,J,K,I] += 0.5 * X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2b[A,D,I,L] * T3abb[a,B,C,i,J,K]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[B,A,C,J,I,K] += X[A,B,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,I,J,K]
R3b[B,C,A,J,K,I] += X[A,B,C,I,J,K]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,i=>x,L=>x,a=>10*x,D=>10*x) begin
X[A,B,C,I,J,K] := oOvV[i,L,a,D] * T2ab[a,A,i,I] * T3b[B,C,D,J,K,L]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,I,J,K]
R3b[B,C,A,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[B,A,C,J,I,K] += X[A,B,C,I,J,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,I,J,K]
R3b[B,C,A,J,K,I] += X[A,B,C,I,J,K]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2ab[d,A,i,I] * T3a[a,b,c,j,k,l]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2ab[a,A,l,I] * T3a[b,c,d,i,j,k]
R3a[a,b,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2a[a,d,i,j] * T3aab[b,c,A,k,l,I]
R3a[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,i,k,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,i,k,j] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,i,k,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,k,i,j] += 0.5 * X[a,b,c,i,j,k]
R3a[b,a,c,k,i,j] -= 0.5 * X[a,b,c,i,j,k]
R3a[b,c,a,k,i,j] += 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2a[a,b,i,l] * T3aab[c,d,A,j,k,I]
R3a[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,j,i,k] -= 0.5 * X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += 0.5 * X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= 0.5 * X[a,b,c,i,j,k]
R3a[c,a,b,j,k,i] += 0.5 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2ab[a,A,i,I] * T3a[b,c,d,j,k,l]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,i,j,k]
R3a[b,c,a,j,k,i] += X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,I=>x,d=>10*x,A=>10*x) begin
X[a,b,c,i,j,k] := oOvV[l,I,d,A] * T2a[a,d,i,l] * T3aab[b,c,A,j,k,I]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[b,a,c,i,j,k] -= X[a,b,c,i,j,k]
R3a[b,c,a,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3a[b,c,a,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[b,a,c,j,k,i] -= X[a,b,c,i,j,k]
R3a[b,c,a,j,k,i] += X[a,b,c,i,j,k]
end
oOvV = nothing
d_oVvV = load4idx(EC,"d_oVvV")
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oVvV[k,A,c,B] * T2a[a,b,i,k] * T2ab[c,B,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oVvV[k,A,c,B] * T2ab[a,B,k,I] * T2a[b,c,i,j]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x,B=>10*x) begin
X[a,b,A,i,j,I] := d_oVvV[k,A,c,B] * T2a[a,c,i,k] * T2ab[b,B,j,I]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oVvV[j,A,b,C] * T2ab[a,B,j,I] * T2ab[b,C,i,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oVvV[j,A,b,C] * T2ab[a,C,j,I] * T2ab[b,B,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oVvV[j,A,b,C] * T2ab[b,B,j,I] * T2ab[a,C,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x,C=>10*x) begin
X[a,A,B,i,I,J] := d_oVvV[j,A,b,C] * T2a[a,b,i,j] * T2b[B,C,I,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,J=>x,I=>x,K=>x,i=>x,a=>10*x,D=>10*x) begin
X[A,B,C,J,I,K] := d_oVvV[i,A,a,D] * T2ab[a,B,i,I] * T2b[C,D,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,J,I,K]
R3b[A,C,B,J,I,K] += X[A,B,C,J,I,K]
R3b[A,B,C,J,K,I] += X[A,B,C,J,I,K]
R3b[A,C,B,J,K,I] -= X[A,B,C,J,I,K]
R3b[B,A,C,J,I,K] += X[A,B,C,J,I,K]
R3b[C,A,B,J,I,K] -= X[A,B,C,J,I,K]
R3b[B,A,C,J,K,I] -= X[A,B,C,J,I,K]
R3b[C,A,B,J,K,I] += X[A,B,C,J,I,K]
R3b[B,C,A,J,I,K] -= X[A,B,C,J,I,K]
R3b[C,B,A,J,I,K] += X[A,B,C,J,I,K]
R3b[B,C,A,J,K,I] += X[A,B,C,J,I,K]
R3b[C,B,A,J,K,I] -= X[A,B,C,J,I,K]
R3b[B,A,C,I,J,K] -= X[A,B,C,J,I,K]
R3b[B,C,A,I,J,K] += X[A,B,C,J,I,K]
R3b[A,B,C,I,J,K] += X[A,B,C,J,I,K]
R3b[C,B,A,I,J,K] -= X[A,B,C,J,I,K]
R3b[A,C,B,I,J,K] -= X[A,B,C,J,I,K]
R3b[C,A,B,I,J,K] += X[A,B,C,J,I,K]
end
d_oVvV = nothing
@mtensoropt begin
X[a,b,c,j,k,l] := fij[i,j] * T3a[a,b,c,k,l,i]
R3a[a,b,c,j,k,l] -= X[a,b,c,j,k,l]
R3a[a,b,c,k,j,l] += X[a,b,c,j,k,l]
R3a[a,b,c,k,l,j] -= X[a,b,c,j,k,l]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := fia[l,d] * T2a[a,b,i,l] * T2a[c,d,j,k]
R3a[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3a[a,c,b,i,j,k] -= X[a,b,c,i,j,k]
R3a[c,a,b,i,j,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,i,k] -= X[a,b,c,i,j,k]
R3a[a,c,b,j,i,k] += X[a,b,c,i,j,k]
R3a[a,b,c,j,k,i] += X[a,b,c,i,j,k]
R3a[a,c,b,j,k,i] -= X[a,b,c,i,j,k]
R3a[c,a,b,j,i,k] -= X[a,b,c,i,j,k]
R3a[c,a,b,j,k,i] += X[a,b,c,i,j,k]
end
@mtensoropt begin
X[b,c,d,i,j,k] := fab[b,a] * T3a[c,d,a,i,j,k]
R3a[b,c,d,i,j,k] += X[b,c,d,i,j,k]
R3a[c,b,d,i,j,k] -= X[b,c,d,i,j,k]
R3a[c,d,b,i,j,k] += X[b,c,d,i,j,k]
end
@mtensoropt begin
X[A,B,C,J,K,L] := fIJ[I,J] * T3b[A,B,C,K,L,I]
R3b[A,B,C,J,K,L] -= X[A,B,C,J,K,L]
R3b[A,B,C,K,J,L] += X[A,B,C,J,K,L]
R3b[A,B,C,K,L,J] -= X[A,B,C,J,K,L]
end
@mtensoropt (A=>10*x,B=>10*x,C=>10*x,I=>x,J=>x,K=>x,L=>x,D=>10*x) begin
X[A,B,C,I,J,K] := fIA[L,D] * T2b[A,B,I,L] * T2b[C,D,J,K]
R3b[A,B,C,I,J,K] += X[A,B,C,I,J,K]
R3b[A,C,B,I,J,K] -= X[A,B,C,I,J,K]
R3b[C,A,B,I,J,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,I,K] -= X[A,B,C,I,J,K]
R3b[A,C,B,J,I,K] += X[A,B,C,I,J,K]
R3b[A,B,C,J,K,I] += X[A,B,C,I,J,K]
R3b[A,C,B,J,K,I] -= X[A,B,C,I,J,K]
R3b[C,A,B,J,I,K] -= X[A,B,C,I,J,K]
R3b[C,A,B,J,K,I] += X[A,B,C,I,J,K]
end
@mtensoropt begin
X[B,C,D,I,J,K] := fAB[B,A] * T3b[C,D,A,I,J,K]
R3b[B,C,D,I,J,K] += X[B,C,D,I,J,K]
R3b[C,B,D,I,J,K] -= X[B,C,D,I,J,K]
R3b[C,D,B,I,J,K] += X[B,C,D,I,J,K]
end
@mtensoropt R3abb[a,A,B,j,I,J] += fij[i,j] * T3abb[a,A,B,i,J,I]
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,j=>x,b=>10*x) begin
X[a,A,B,i,I,J] := fia[j,b] * T2ab[a,A,j,I] * T2ab[b,B,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,J,I] -= X[a,A,B,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := fIA[K,C] * T2ab[a,A,i,K] * T2b[B,C,I,J]
R3abb[a,A,B,i,I,J] += X[a,A,B,i,I,J]
R3abb[a,B,A,i,I,J] -= X[a,A,B,i,I,J]
end
@mtensoropt R3abb[b,A,B,i,I,J] -= fab[b,a] * T3abb[a,B,A,i,I,J]
@mtensoropt begin
X[a,B,C,i,I,J] := fAB[B,A] * T3abb[a,C,A,i,I,J]
R3abb[a,B,C,i,I,J] -= X[a,B,C,i,I,J]
R3abb[a,C,B,i,I,J] += X[a,B,C,i,I,J]
end
@mtensoropt (a=>10*x,A=>10*x,B=>10*x,i=>x,I=>x,J=>x,K=>x,C=>10*x) begin
X[a,A,B,i,I,J] := fIA[K,C] * T2b[A,B,I,K] * T2ab[a,C,i,J]
R3abb[a,A,B,i,I,J] -= X[a,A,B,i,I,J]
R3abb[a,A,B,i,J,I] += X[a,A,B,i,I,J]
end
@mtensoropt begin
X[a,A,B,i,J,K] := fIJ[I,J] * T3abb[a,A,B,i,K,I]
R3abb[a,A,B,i,J,K] += X[a,A,B,i,J,K]
R3abb[a,A,B,i,K,J] -= X[a,A,B,i,J,K]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := fia[k,c] * T2ab[a,A,k,I] * T2a[b,c,i,j]
R3aab[a,b,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] -= X[a,b,A,i,j,I]
end
@mtensoropt R3aab[a,b,A,i,j,J] += fIJ[I,J] * T3aab[a,b,A,j,i,I]
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,J=>x,B=>10*x) begin
X[a,b,A,i,j,I] := fIA[J,B] * T2ab[a,A,i,J] * T2ab[b,B,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[b,a,A,i,j,I] += X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
R3aab[b,a,A,j,i,I] -= X[a,b,A,i,j,I]
end
@mtensoropt R3aab[a,b,B,i,j,I] -= fAB[B,A] * T3aab[b,a,A,i,j,I]
@mtensoropt begin
X[b,c,A,i,j,I] := fab[b,a] * T3aab[c,a,A,i,j,I]
R3aab[b,c,A,i,j,I] -= X[b,c,A,i,j,I]
R3aab[c,b,A,i,j,I] += X[b,c,A,i,j,I]
end
@mtensoropt (a=>10*x,b=>10*x,A=>10*x,i=>x,j=>x,I=>x,k=>x,c=>10*x) begin
X[a,b,A,i,j,I] := fia[k,c] * T2a[a,b,i,k] * T2ab[c,A,j,I]
R3aab[a,b,A,i,j,I] -= X[a,b,A,i,j,I]
R3aab[a,b,A,j,i,I] += X[a,b,A,i,j,I]
end
@mtensoropt begin
X[a,b,A,j,k,I] := fij[i,j] * T3aab[a,b,A,k,i,I]
R3aab[a,b,A,j,k,I] += X[a,b,A,j,k,I]
R3aab[a,b,A,k,j,I] -= X[a,b,A,j,k,I]
end
end

function dcccsdt_triples!(EC::ECInfo, R3, T2, T3, fij, fab, fai, fia)
#bract
#act,divide=$(1 - \Perm{abc}{cab})$
# d_vvvv = load4idx(EC,"d_vvvv")
# @mtensoropt R3[e,c,d,i,j,k] += d_vvvv[c,d,a,b] * T3[a,e,b,j,i,k]
# @mtensoropt R3[c,e,d,i,j,k] += d_vvvv[c,d,a,b] * T3[a,e,b,i,j,k]
# @mtensoropt R3[c,d,e,i,j,k] += d_vvvv[c,d,a,b] * T3[a,e,b,i,k,j]
# d_vvvv = nothing
triples_4ext!(EC, R3, T3)
d_vvvo = load4idx(EC,"d_vvvo")
@mtensoropt begin
X[b,c,d,i,j,k] := d_vvvo[c,b,a,i] * T2[a,d,j,k]
R3[b,c,d,i,j,k] += X[b,c,d,i,j,k]
R3[b,d,c,i,j,k] += X[b,c,d,i,k,j]
R3[b,c,d,j,i,k] += X[c,b,d,i,j,k]
R3[d,b,c,j,i,k] += X[b,c,d,i,k,j]
R3[b,d,c,j,k,i] += X[c,b,d,i,j,k]
R3[d,b,c,j,k,i] += X[c,b,d,i,k,j]
end
d_vvvo = nothing
d_vovv = load4idx(EC,"d_vovv")
@mtensoropt (c=>10*x,a=>10*x,b=>10*x,j=>x,k=>x,i=>x,l=>x,d=>10*x,e=>10*x) begin
X[c,a,b,j,k,i] := d_vovv[a,l,d,e] * T2[d,b,l,i] * T2[e,c,k,j]
R3[c,a,b,j,k,i] -= X[c,a,b,j,k,i]
R3[c,b,a,j,i,k] -= X[c,a,b,j,k,i]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,k=>x,i=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,j,k,i] := d_vovv[a,l,e,d] * T2[b,c,l,i] * T2[d,e,k,j]
R3[a,b,c,j,k,i] -= X[a,b,c,j,k,i]
R3[a,b,c,j,i,k] -= X[a,c,b,j,k,i]
R3[b,a,c,j,k,i] -= X[a,b,c,k,j,i]
R3[b,a,c,i,j,k] -= X[a,c,b,j,k,i]
R3[b,c,a,j,i,k] -= X[a,b,c,k,j,i]
R3[b,c,a,i,j,k] -= X[a,c,b,k,j,i]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,i,j,k] := d_vovv[a,l,d,e] * T2[b,d,l,i] * T2[e,c,j,k]
R3[a,b,c,i,j,k] -= X[a,b,c,i,j,k]
R3[a,c,b,i,j,k] -= X[a,b,c,i,k,j]
R3[b,a,c,j,i,k] -= X[a,b,c,i,j,k]
R3[c,a,b,j,i,k] -= X[a,b,c,i,k,j]
R3[b,c,a,j,k,i] -= X[a,b,c,i,j,k]
R3[c,b,a,j,k,i] -= X[a,b,c,i,k,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,d,e] * T2[d,b,l,i] * T2[e,c,j,k]
R3[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3[b,a,c,i,j,k] -= X[a,b,c,j,i,k]
R3[b,c,a,i,j,k] -= X[a,b,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,e,d] * T2[b,d,l,i] * T2[e,c,j,k]
R3[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3[b,a,c,i,j,k] -= X[a,b,c,j,i,k]
R3[c,a,b,j,k,i] -= X[a,b,c,k,i,j]
R3[b,c,a,i,j,k] -= X[a,b,c,k,i,j]
R3[c,b,a,j,i,k] -= X[a,b,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,j,i,k] := d_vovv[a,l,e,d] * T2[d,b,l,i] * T2[e,c,j,k]
R3[a,b,c,j,i,k] += 2 * X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] += 2 * X[a,b,c,j,i,k]
R3[b,a,c,i,j,k] += 2 * X[a,b,c,j,i,k]
R3[b,c,a,i,j,k] += 2 * X[a,b,c,k,i,j]
R3[c,a,b,j,k,i] += 2 * X[a,b,c,k,i,j]
R3[c,b,a,j,i,k] += 2 * X[a,b,c,k,i,j]
end
d_vovv = nothing
d_vovo = load4idx(EC,"d_vovo")
@mtensoropt begin
X[c,b,d,j,k,l] := d_vovo[b,i,a,j] * T3[c,a,d,i,k,l]
R3[c,b,d,j,k,l] -= X[c,b,d,j,k,l]
R3[c,d,b,j,k,l] -= X[c,b,d,j,l,k]
R3[b,c,d,k,j,l] -= X[c,b,d,j,k,l]
R3[c,d,b,k,j,l] -= X[d,b,c,j,l,k]
R3[b,c,d,k,l,j] -= X[d,b,c,j,k,l]
R3[c,b,d,k,l,j] -= X[d,b,c,j,l,k]
end
@mtensoropt begin
X[b,c,d,j,k,l] := d_vovo[b,i,a,j] * T3[a,c,d,i,k,l]
R3[b,c,d,j,k,l] -= X[b,c,d,j,k,l]
R3[c,b,d,k,j,l] -= X[b,c,d,j,k,l]
R3[c,d,b,k,l,j] -= X[b,c,d,j,k,l]
end
d_vovo = nothing
d_voov = load4idx(EC,"d_voov")
@mtensoropt begin
X[b,c,d,j,k,l] := d_voov[b,i,j,a] * T3[c,a,d,i,k,l]
R3[b,c,d,j,k,l] -= X[b,c,d,j,k,l]
R3[b,c,d,j,k,l] -= X[b,d,c,j,l,k]
R3[c,b,d,k,j,l] -= X[b,c,d,j,k,l]
R3[c,b,d,k,j,l] -= X[b,d,c,j,l,k]
R3[c,d,b,k,l,j] -= X[b,c,d,j,k,l]
R3[c,d,b,k,l,j] -= X[b,d,c,j,l,k]
end
@mtensoropt begin
X[b,c,d,j,k,l] := d_voov[b,i,j,a] * T3[a,c,d,i,k,l]
R3[b,c,d,j,k,l] += 2 * X[b,c,d,j,k,l]
R3[c,b,d,k,j,l] += 2 * X[b,c,d,j,k,l]
R3[c,d,b,k,l,j] += 2 * X[b,c,d,j,k,l]
end
d_voov = nothing
d_vooo = load4idx(EC,"d_vooo")
@mtensoropt begin
X[b,a,c,j,k,l] := d_vooo[a,i,k,j] * T2[b,c,i,l]
R3[b,a,c,j,k,l] -= X[b,a,c,j,k,l]
R3[b,c,a,j,l,k] -= X[b,a,c,j,k,l]
R3[a,b,c,j,k,l] -= X[b,a,c,k,j,l]
R3[b,c,a,l,j,k] -= X[c,a,b,j,k,l]
R3[a,b,c,j,l,k] -= X[c,a,b,k,j,l]
R3[b,a,c,l,j,k] -= X[c,a,b,k,j,l]
end
d_vooo = nothing
d_oooo = load4idx(EC,"d_oooo")
@mtensoropt begin
X[a,b,c,k,l,m] := d_oooo[j,i,l,k] * T3[a,b,c,i,j,m]
R3[a,b,c,k,l,m] += X[a,b,c,k,l,m]
R3[a,b,c,k,m,l] += X[a,c,b,k,l,m]
R3[a,b,c,m,k,l] += X[b,c,a,k,l,m]
end
d_oooo = nothing
d_oovo = load4idx(EC,"d_oovo")
@mtensoropt (b=>10*x,c=>10*x,a=>10*x,k=>x,i=>x,j=>x,m=>x,l=>x,d=>10*x) begin
X[b,c,a,k,i,j] := d_oovo[m,l,d,i] * T2[d,a,l,j] * T2[c,b,m,k]
R3[b,c,a,k,i,j] += X[b,c,a,k,i,j]
R3[b,a,c,k,j,i] += X[b,c,a,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2[a,b,l,j] * T2[c,d,m,k]
R3[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3[a,c,b,i,k,j] += X[a,b,c,i,j,k]
R3[a,b,c,j,i,k] += X[b,a,c,i,j,k]
R3[c,a,b,k,i,j] += X[a,b,c,i,j,k]
R3[a,c,b,j,k,i] += X[b,a,c,i,j,k]
R3[c,a,b,k,j,i] += X[b,a,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2[a,d,l,j] * T2[b,c,m,k]
R3[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3[a,b,c,i,k,j] += X[a,c,b,i,j,k]
R3[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3[b,a,c,k,i,j] += X[a,c,b,i,j,k]
R3[b,c,a,j,k,i] += X[a,b,c,i,j,k]
R3[b,c,a,k,j,i] += X[a,c,b,i,j,k]
end
@mtensoropt (b=>10*x,a=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[b,a,c,i,j,k] := d_oovo[m,l,d,i] * T2[d,a,l,j] * T2[b,c,m,k]
R3[b,a,c,i,j,k] += X[b,a,c,i,j,k]
R3[b,c,a,i,k,j] += X[b,a,c,i,j,k]
R3[a,b,c,j,i,k] += X[b,a,c,i,j,k]
R3[a,b,c,j,k,i] += X[c,a,b,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2[a,b,l,m] * T2[d,c,j,k]
R3[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3[a,c,b,i,j,k] += X[a,b,c,i,k,j]
R3[a,b,c,j,i,k] += X[b,a,c,i,j,k]
R3[c,a,b,j,i,k] += X[a,b,c,i,k,j]
R3[a,c,b,j,k,i] += X[b,a,c,i,j,k]
R3[c,a,b,j,k,i] += X[b,a,c,i,k,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x) begin
X[a,b,c,i,j,k] := d_oovo[m,l,d,i] * T2[a,b,l,j] * T2[d,c,m,k]
R3[a,b,c,i,j,k] -= 2 * X[a,b,c,i,j,k]
R3[a,c,b,i,k,j] -= 2 * X[a,b,c,i,j,k]
R3[a,b,c,j,i,k] -= 2 * X[b,a,c,i,j,k]
R3[a,c,b,j,k,i] -= 2 * X[b,a,c,i,j,k]
R3[c,a,b,k,i,j] -= 2 * X[a,b,c,i,j,k]
R3[c,a,b,k,j,i] -= 2 * X[b,a,c,i,j,k]
end
d_oovo = nothing
@mtensoropt begin
X[a,b,c,j,k,l] := fij[i,j] * T3[a,b,c,i,k,l]
R3[a,b,c,j,k,l] -= X[a,b,c,j,k,l]
R3[a,b,c,k,j,l] -= X[b,a,c,j,k,l]
R3[a,b,c,k,l,j] -= X[c,a,b,j,k,l]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,l=>x,d=>10*x) begin
X[a,b,c,j,i,k] := fia[l,d] * T2[a,b,l,i] * T2[d,c,j,k]
R3[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3[a,b,c,i,j,k] -= X[b,a,c,j,i,k]
R3[a,c,b,i,j,k] -= X[b,a,c,k,i,j]
R3[c,a,b,j,k,i] -= X[a,b,c,k,i,j]
R3[c,a,b,j,i,k] -= X[b,a,c,k,i,j]
end
@mtensoropt begin
X[b,c,d,i,j,k] := fab[b,a] * T3[a,c,d,i,j,k]
R3[b,c,d,i,j,k] += X[b,c,d,i,j,k]
R3[c,b,d,i,j,k] += X[b,c,d,j,i,k]
R3[c,d,b,i,j,k] += X[b,c,d,k,i,j]
end
oovv = ints2(EC,"oovv")
@mtensoropt (b=>10*x,a=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,m=>x,e=>10*x,d=>10*x) begin
X[b,a,c,i,j,k] := oovv[l,m,e,d] * T2[d,a,i,j] * T3[e,b,c,m,l,k]
R3[b,a,c,i,j,k] += 0.5 * X[b,a,c,i,j,k]
R3[b,c,a,i,k,j] += 0.5 * X[b,a,c,i,j,k]
R3[a,b,c,i,j,k] += 0.5 * X[b,a,c,j,i,k]
R3[b,c,a,k,i,j] += 0.5 * X[c,a,b,i,j,k]
R3[a,b,c,i,k,j] += 0.5 * X[c,a,b,j,i,k]
R3[b,a,c,k,i,j] += 0.5 * X[c,a,b,j,i,k]
end
@mtensoropt (b=>10*x,a=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,m=>x,e=>10*x,d=>10*x) begin
X[b,a,c,i,j,k] := oovv[l,m,e,d] * T2[d,a,i,j] * T3[b,c,e,m,l,k]
R3[b,a,c,i,j,k] += 0.5 * X[b,a,c,i,j,k]
R3[b,c,a,i,k,j] += 0.5 * X[b,a,c,i,j,k]
R3[a,b,c,i,j,k] += 0.5 * X[b,a,c,j,i,k]
R3[b,c,a,k,i,j] += 0.5 * X[c,a,b,i,j,k]
R3[a,b,c,i,k,j] += 0.5 * X[c,a,b,j,i,k]
R3[b,a,c,k,i,j] += 0.5 * X[c,a,b,j,i,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,m=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,j,i,k] := oovv[m,l,d,e] * T2[a,b,l,i] * T3[e,d,c,m,j,k]
R3[a,b,c,j,i,k] += 0.5 * X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] += 0.5 * X[a,b,c,j,i,k]
R3[a,b,c,i,j,k] += 0.5 * X[b,a,c,j,i,k]
R3[c,a,b,j,k,i] += 0.5 * X[a,b,c,k,i,j]
R3[a,c,b,i,j,k] += 0.5 * X[b,a,c,k,i,j]
R3[c,a,b,j,i,k] += 0.5 * X[b,a,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,m=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,j,i,k] := oovv[m,l,d,e] * T2[a,b,l,i] * T3[e,c,d,j,m,k]
R3[a,b,c,j,i,k] += 0.5 * X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] += 0.5 * X[a,b,c,j,i,k]
R3[a,b,c,i,j,k] += 0.5 * X[b,a,c,j,i,k]
R3[c,a,b,j,k,i] += 0.5 * X[a,b,c,k,i,j]
R3[a,c,b,i,j,k] += 0.5 * X[b,a,c,k,i,j]
R3[c,a,b,j,i,k] += 0.5 * X[b,a,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[a,d,l,i] * T3[b,e,c,m,j,k]
R3[a,b,c,i,j,k] += X[a,b,c,i,j,k]
R3[a,b,c,i,j,k] += X[a,c,b,i,k,j]
R3[b,a,c,j,i,k] += X[a,b,c,i,j,k]
R3[b,a,c,j,i,k] += X[a,c,b,i,k,j]
R3[b,c,a,j,k,i] += X[a,b,c,i,j,k]
R3[b,c,a,j,k,i] += X[a,c,b,i,k,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,d,e] * T2[d,e,l,i] * T3[a,b,c,m,j,k]
R3[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3[a,b,c,j,i,k] += 0.5 * X[b,a,c,i,j,k]
R3[a,b,c,j,k,i] += 0.5 * X[c,a,b,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[a,d,l,m] * T3[e,b,c,i,j,k]
R3[a,b,c,i,j,k] += 0.5 * X[a,b,c,i,j,k]
R3[b,a,c,i,j,k] += 0.5 * X[a,b,c,j,i,k]
R3[b,c,a,i,j,k] += 0.5 * X[a,b,c,k,i,j]
end
@mtensoropt (b=>10*x,a=>10*x,c=>10*x,i=>x,j=>x,k=>x,l=>x,m=>x,e=>10*x,d=>10*x) begin
X[b,a,c,i,j,k] := oovv[l,m,e,d] * T2[d,a,i,j] * T3[b,e,c,m,l,k]
R3[b,a,c,i,j,k] -= X[b,a,c,i,j,k]
R3[b,c,a,i,k,j] -= X[b,a,c,i,j,k]
R3[a,b,c,i,j,k] -= X[b,a,c,j,i,k]
R3[b,c,a,k,i,j] -= X[c,a,b,i,j,k]
R3[a,b,c,i,k,j] -= X[c,a,b,j,i,k]
R3[b,a,c,k,i,j] -= X[c,a,b,j,i,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,j=>x,i=>x,k=>x,m=>x,l=>x,d=>10*x,e=>10*x) begin
X[a,b,c,j,i,k] := oovv[m,l,d,e] * T2[a,b,l,i] * T3[e,d,c,j,m,k]
R3[a,b,c,j,i,k] -= X[a,b,c,j,i,k]
R3[a,c,b,j,k,i] -= X[a,b,c,j,i,k]
R3[a,b,c,i,j,k] -= X[b,a,c,j,i,k]
R3[c,a,b,j,k,i] -= X[a,b,c,k,i,j]
R3[a,c,b,i,j,k] -= X[b,a,c,k,i,j]
R3[c,a,b,j,i,k] -= X[b,a,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[a,d,l,i] * T3[e,b,c,m,j,k]
R3[a,b,c,i,j,k] -= 2 * X[a,b,c,i,j,k]
R3[b,a,c,j,i,k] -= 2 * X[a,b,c,i,j,k]
R3[b,c,a,j,k,i] -= 2 * X[a,b,c,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[d,a,l,i] * T3[b,e,c,m,j,k]
R3[a,b,c,i,j,k] -= 2 * X[a,b,c,i,j,k]
R3[a,b,c,i,j,k] -= 2 * X[a,c,b,i,k,j]
R3[b,a,c,j,i,k] -= 2 * X[a,b,c,i,j,k]
R3[b,a,c,j,i,k] -= 2 * X[a,c,b,i,k,j]
R3[b,c,a,j,k,i] -= 2 * X[a,b,c,i,j,k]
R3[b,c,a,j,k,i] -= 2 * X[a,c,b,i,k,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[d,e,l,i] * T3[a,b,c,m,j,k]
R3[a,b,c,i,j,k] -= X[a,b,c,i,j,k]
R3[a,b,c,j,i,k] -= X[b,a,c,i,j,k]
R3[a,b,c,j,k,i] -= X[c,a,b,i,j,k]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[d,a,l,m] * T3[e,b,c,i,j,k]
R3[a,b,c,i,j,k] -= X[a,b,c,i,j,k]
R3[b,a,c,i,j,k] -= X[a,b,c,j,i,k]
R3[b,c,a,i,j,k] -= X[a,b,c,k,i,j]
end
@mtensoropt (a=>10*x,b=>10*x,c=>10*x,i=>x,j=>x,k=>x,m=>x,l=>x,e=>10*x,d=>10*x) begin
X[a,b,c,i,j,k] := oovv[m,l,e,d] * T2[d,a,l,i] * T3[e,b,c,m,j,k]
R3[a,b,c,i,j,k] += 4 * X[a,b,c,i,j,k]
R3[b,a,c,j,i,k] += 4 * X[a,b,c,i,j,k]
R3[b,c,a,j,k,i] += 4 * X[a,b,c,i,j,k]
end
oovv = nothing
end
