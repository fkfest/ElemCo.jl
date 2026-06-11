using ElemCo
using ElemCo.ECInfos
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx, overlap, kinetic, nuclear
using ElemCo.IntegralTools: generate_ao_fdump, transform_ao2mo
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.TensorTools: detri_int2
using LinearAlgebra

@testset "AO-FDump (non-DF AO integrals)" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"

  EC = ECInfo{Float64}()
  EC.system = parse_geometry(geometry, Dict("ao"=>"sto-3g"))

  bao = generate_basis(EC, "ao")
  nao = n_ao(bao)
  G = eri_2e4idx(bao)          # chemists' (μν|ρσ)
  S = overlap(bao)
  hAO = kinetic(bao) + nuclear(bao)

  fd = generate_ao_fdump(EC)

  @testset "flags and metadata" begin
    @test is_ao_basis(fd)
    @test fd.ao_basis
    @test headvar(fd, "AOBASIS", Int) == 1
    @test headvar(fd, "NORB", Int) == nao
    @test size(fd.overlap) == (nao, nao)
    @test maximum(abs.(fd.overlap .- S)) < 1e-12
    @test maximum(abs.(fd.int1 .- hAO)) < 1e-12
  end

  @testset "physicist-notation round-trip" begin
    # detri reconstructs full <pq|rs> from the triangular storage;
    # it must equal (pr|qs) = G[p,r,q,s]
    sp = 1:nao
    v = detri_int2(fd.int2, nao, sp, sp, sp, sp)   # v[p,q,r,s] = <pq|rs>
    Gphys = permutedims(G, (1,3,2,4))              # G[p,r,q,s] -> [p,q,r,s]
    @test maximum(abs.(v .- Gphys)) < 1e-12
  end

  @testset "AO->MO transform (non-DF)" begin
    # contract dimension `dim` of the 4-tensor A with the matrix X (A_idx -> X[idx,new])
    function reduce_index(A, X, dim)
      perm = (dim, setdiff(1:4, dim)...)
      Ap = permutedims(A, perm)                       # transformed axis first
      n = size(Ap, 1)
      M = X' * reshape(Ap, n, :)                      # (norb, rest)
      Bp = reshape(M, size(X, 2), size(Ap, 2), size(Ap, 3), size(Ap, 4))
      return permutedims(Bp, invperm(collect(perm)))  # restore axis order
    end

    # symmetric orthogonalization: X' S X = I  (use X as MO coefficients)
    F = eigen(Symmetric(S))
    X = F.vectors * Diagonal(1.0 ./ sqrt.(F.values)) * F.vectors'
    fd_mo = transform_ao2mo(fd, X)

    # independent reference: naive chemist-basis 4-index transform, then -> physicist
    Gmo = reduce_index(reduce_index(reduce_index(reduce_index(G, X, 1), X, 2), X, 3), X, 4)
    vmo_ref = permutedims(Gmo, (1,3,2,4))   # chemist (pr|qs) -> physicist <pq|rs>

    spm = 1:nao
    vmo = detri_int2(fd_mo.int2, nao, spm, spm, spm, spm)
    @test maximum(abs.(vmo .- vmo_ref)) < 1e-10
    # MO 1-e integrals
    @test maximum(abs.(fd_mo.int1 .- X' * hAO * X)) < 1e-10
    # full 8-fold symmetry of the real MO integrals
    @test maximum(abs.(vmo .- permutedims(vmo, (3,4,1,2)))) < 1e-10
    # not flagged AO any more
    @test !fd_mo.ao_basis
    @test abs(fd_mo.int0 - fd.int0) < 1e-12
  end

  @testset "exact AO-HF vs reference RHF" begin
    # independent closed-shell RHF from S, h, and the exact chemist tensor G
    function ref_rhf(S, h, G, Enuc, nocc; maxit=200, thr=1e-11)
      nb = size(S, 1)
      Fe = eigen(Symmetric(S))
      X = Fe.vectors * Diagonal(1.0 ./ sqrt.(Fe.values)) * Fe.vectors'
      ev, C = eigen(Symmetric(X' * h * X)); Cmo = X * C
      E = 0.0
      for _ in 1:maxit
        Cocc = Cmo[:, 1:nocc]
        D = Cocc * Cocc'
        J = zeros(nb, nb); K = zeros(nb, nb)
        @inbounds for q in 1:nb, p in 1:nb
          jpq = 0.0; kpq = 0.0
          for s in 1:nb, r in 1:nb
            jpq += G[p, q, r, s] * D[r, s]   # (pq|rs)
            kpq += G[p, s, r, q] * D[r, s]   # (ps|rq)
          end
          J[p, q] = jpq; K[p, q] = kpq
        end
        F = h + 2 .* J .- K
        Enew = sum(D .* (h .+ F)) + Enuc
        ev, C = eigen(Symmetric(X' * F * X)); Cmo = X * C
        if abs(Enew - E) < thr
          E = Enew; break
        end
        E = Enew
      end
      return E
    end

    nelec = headvar(fd, "NELEC", Int)
    Eref = ref_rhf(S, hAO, G, fd.int0, nelec ÷ 2)

    EC2 = ECInfo{Float64}()
    EC2.system = parse_geometry(geometry, Dict("ao"=>"sto-3g"))
    EC2.fd = generate_ao_fdump(EC2)
    res = ElemCo.ao_hf(EC2)
    @test abs(res["HF"] - Eref) < 1e-7
  end

  @testset "npy write/read round-trip" begin
    tmp = mktempdir()
    path = joinpath(tmp, "AO.FCIDUMP")
    write_fcidump(fd, path; format=:npy)
    fd2 = read_fcidump(path, Float64)
    @test is_ao_basis(fd2)
    @test headvar(fd2, "NORB", Int) == nao
    @test maximum(abs.(fd2.int2 .- fd.int2)) < 1e-12
    @test maximum(abs.(fd2.int1 .- fd.int1)) < 1e-12
    @test maximum(abs.(fd2.overlap .- fd.overlap)) < 1e-12
    @test abs(fd2.int0 - fd.int0) < 1e-12
    rm(tmp; recursive=true)
  end
end
