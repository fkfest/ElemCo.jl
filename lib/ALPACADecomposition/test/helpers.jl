using TestItems

@testsnippet Helpers begin
  using ALPACADecomposition
  using LinearAlgebra

  function reconstruct(result::ALPACAResult{T}) where T
    L = result.left
    R = result.right
    sym = result.symmetry

    if sym == :general
      return L * R'
    else
      k = size(L, 2)
      S = ones(real(T), k)
      for i in result.neg_indices
        S[i] = -one(real(T))
      end
      if sym == :hermitian
        return L * Diagonal(S) * L'
      else
        return L * Diagonal(S) * transpose(L)
      end
    end
  end
end
