# Functions for upper triangular tensors stored as [p,q,...,tri] arrays


"""
    lentri_from_norb(n)

  Return the length of the upper triangular part of a tensor of dimension n×n.
"""
lentri_from_norb(n) = n*(n+1)÷2

"""
    lentri_from_norb(n, N)

  Return the length of the upper triangular part of a tensor of dimension n^N.
"""
lentri_from_norb(n, N) = prod(n:n+N-1)÷factorial(N)

"""
    norb_from_lentri(tri2)

  Return the number of orbitals from the length of triangular index `tri` (for dimension n×n).
"""
norb_from_lentri(tri2) = Int(sqrt(8*tri2+1)-1)÷2

"""
    norb_from_lentri(triN, N)

  Return the number of orbitals from the triangular index of size `triN`.
"""
function norb_from_lentri(triN, N)
  n = trunc(Int, (triN * factorial(N))^(1/N)) - (N)÷2 + 1
  @assert lentri_from_norb(n, N) == triN "The dimension $triN is not triangular of $N×$n."
  return n
end

"""
    strict_lentri_from_norb(n)

  Return the length of the strict upper triangular part of a tensor of dimension n×n.
"""
strict_lentri_from_norb(n) = n*(n-1)÷2

"""
    strict_lentri_from_norb(n, N)

  Return the length of the strict upper triangular part of a tensor of dimension n^N.
"""
strict_lentri_from_norb(n, N) = prod(n-N+1:n)÷factorial(N)

"""
    norb_from_strict_lentri(tri2)

  Return the number of orbitals from the length of strict triangular index `tri2` (for dimension n×n).
"""
norb_from_strict_lentri(tri2) = Int(sqrt(8*tri2+1)+1)÷2

"""
    norb_from_strict_lentri(triN, N)

  Return the number of orbitals from the length of strict triangular index of size `triN`.
"""
function norb_from_strict_lentri(triN, N)
  n = trunc(Int, (triN * factorial(N))^(1/N)) + (N+1)÷2
  @assert strict_lentri_from_norb(n, N) == triN "The dimension $triN is not strict triangular of $n^$N."
  return n
end

"""
    uppertriangular_index(i1, i2)

  Return uppertriangular index from two indices `i1 <= i2`.
"""
function uppertriangular_index(i1, i2)
  @assert i1 <= i2 "The indices are not in the correct order."
  return i1 + i2*(i2-1)÷2
end

""" 
    uppertriangular_index(i1, i2, i3)

  Return uppertriangular index from three indices `i1 <= i2 <= i3`.
"""
function uppertriangular_index(i1, i2, i3)
  return i1 + i2*(i2-1)÷2 + (i3+1)*i3*(i3-1)÷6
end

"""
    uppertriangular_index(inds::Vararg{Int, N})

  Return uppertriangular index from a set of indices `i1 <= i2 <= ... <= iN`.
"""
function uppertriangular_index(inds::Vararg{Int, N}) where N 
  tri = inds[1]
  for i in 2:N
    @assert inds[i-1] <= inds[i] "The indices are not in the correct order."
    tri += lentri_from_norb(inds[i]-1, i)
  end
  return tri
end

""" 
    uppertriangular_range(i2)

  Return range for the uppertriangular index (`i1 <= i2`) for a given `i2`. 
"""
function uppertriangular_range(i2)
  start = i2*(i2-1)÷2+1
  stop = start + i2 - 1
  return start:stop
end

""" 
    uppertriangular_range(inds::Vararg{Int, N}) where N

  Return range for the uppertriangular index (`i1 <= i2 <= i3 <= ...`) for given `i2`, `i3`, ... 
"""
function uppertriangular_range(inds::Vararg{Int, N}) where N
  start = uppertriangular_index(1, inds...)
  stop = start + inds[1] - 1
  return start:stop
end

""" 
    strict_uppertriangular_range(i2)

  Return range for the uppertriangular index (i1 <= i2) without diagonal (i1 < i2) for a given i2. 
"""
function strict_uppertriangular_range(i2)
  start = i2*(i2-1)÷2+1
  stop = start + i2 - 2
  return start:stop
end

""" 
    strict_uppertriangular_range(inds::Vararg{Int, N}) where N

  Return range for the uppertriangular index (`i1 <= i2 <= i3 <= ...`) without diagonal (i1 < i2 <= i3 <= ...)
  for given `i2`, `i3`, ... 
"""
function strict_uppertriangular_range(inds::Vararg{Int, N}) where N 
  start = uppertriangular_index(1, inds...)
  stop = start + inds[1] - 2
  return start:stop
end

"""
    uppertriangular_cut(norb)

  Return all indices for original dimension `norb×norb` corresponding to the upper triangular part.
"""
uppertriangular_cut(norb) = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]

"""
    swapped_uppertriangular_cut(norb)

  Return all indices for original dimension `norb×norb` corresponding to the upper triangular part,
  but with the two indices swapped, i.e., (j,i) instead of (i,j).
"""
swapped_uppertriangular_cut(norb) = [CartesianIndex(j,i) for j in 1:norb for i in 1:j]

"""
    uppertriangular_cut3(norb)

  Return all indices for original dimension `norb×norb×norb` corresponding to the upper triangular part.
"""
uppertriangular_cut3(norb) = [CartesianIndex(i,j,k) for k in 1:norb for j in 1:k for i in 1:j]

"""
    strict_uppertriangular_cut(norb)

  Return all indices for original dimension `norb×norb` corresponding to the strict upper triangular part.
"""
strict_uppertriangular_cut(norb) = [CartesianIndex(i,j) for j in 2:norb for i in 1:j-1]

"""
    swapped_strict_uppertriangular_cut(norb)

  Return all indices for original dimension `norb×norb` corresponding to the strict upper triangular part,
  but with the two indices swapped, i.e., (j,i) instead of (i,j).
"""
swapped_strict_uppertriangular_cut(norb) = [CartesianIndex(j,i) for j in 2:norb for i in 1:j-1]

"""
    strict_uppertriangular_cut3(norb)

  Return all indices for original dimension `norb×norb×norb` corresponding to the strict upper triangular part.
"""
strict_uppertriangular_cut3(norb) = [CartesianIndex(i,j,k) for k in 3:norb for j in 2:k-1 for i in 1:j-1]

"""
    detri_doubles(T2)

  Convert a doubles amplitude tensor T2 in the form (a,b,ij) to the full form (a,b,i,j).
  Here, `ij` is the upper triangular index for occupied orbitals `i <= j`.
"""
function detri_doubles(T2)
  a,b,ij = size(T2)
  nocc = norb_from_lentri(ij)
  T2full = Array{eltype(T2)}(undef, a, b, nocc, nocc)
  tripp = uppertriangular_cut(nocc)
  T2full[:,:,tripp] = T2
  swtripp = swapped_uppertriangular_cut(nocc)
  permutedims!(@view(T2full[:,:,swtripp]), T2, (2, 1, 3))
  return T2full
end

"""
    detri_samespin_doubles(T2::AbstractMatrix{T})

  Convert a doubles amplitude tensor T2 in the form (ab,ij) to the full form (a,b,i,j)
  using the permutational symmetry ``T^{ij}_{ab} = T^{ji}_{ba} = -T^{ij}_{ba} = -T^{ji}_{ab}``.

Here, `ab` and `ij` are the strict upper triangular indices for virtual and occupied orbitals `a < b`, `i < j`.
"""
function detri_samespin_doubles(T2::AbstractMatrix{T}) where T
  ab,ij = size(T2)
  nvir = norb_from_strict_lentri(ab)
  nocc = norb_from_strict_lentri(ij)
  T2full = zeros(T, nvir, nvir, nocc, nocc)
  trioo = strict_uppertriangular_cut(nocc)
  trivv = strict_uppertriangular_cut(nvir)
  swtrioo = swapped_strict_uppertriangular_cut(nocc)
  swtrivv = swapped_strict_uppertriangular_cut(nvir)
  T2full[trivv,trioo] = T2
  T2full[swtrivv,trioo] = -T2
  T2full[trivv,swtrioo] = -T2
  T2full[swtrivv,swtrioo] = T2
  return T2full
end

"""
    calc_tri_sym_antisym!(out_s, out_a, A)

  Compute symmetric and antisymmetric combinations of a 3-index array `A[p,q,x]`
  in a single pass over the data.
  
  ``out\\_s[pq,x] = A[p,q,x] + A[q,p,x]``  (symmetric in p,q)

  ``out\\_a[pq,x] = A[p,q,x] - A[q,p,x]``  (antisymmetric in p,q)
  
  where `pq` is the upper triangular index for `p ≤ q`.

  For each column `q`, the strided row `A[q, 1:q, x]` is copied into a small
  contiguous buffer, then sum/difference is computed with stride-1 SIMD access.
  Multi-threaded over `x`.
"""
function calc_tri_sym_antisym!(out_s::AbstractMatrix, out_a::AbstractMatrix,
                               A::AbstractArray{T,3}) where T
  norb = size(A, 1)
  nx = size(A, 3)
  @threadsbuffer tbuf(norb) begin
  Threads.@threads for x in 1:nx
    buf = alloc!(tbuf, norb)
    @inbounds for q in 1:norb
      pq0 = q * (q - 1) ÷ 2
      @simd for p in 1:q
        buf[p] = A[q, p, x]
      end
      @simd ivdep for p in 1:q-1
        out_s[pq0 + p, x] = A[p, q, x] + buf[p]
        out_a[pq0 + p, x] = A[p, q, x] - buf[p]
      end
      out_s[pq0 + q, x] = buf[q] + buf[q]
      out_a[pq0 + q, x] = zero(eltype(A))
    end
    reset!(tbuf) # reset buffer for the next thread iteration
  end
  end # buffer 
end
