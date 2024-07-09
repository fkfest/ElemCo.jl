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