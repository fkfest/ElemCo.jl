@testitem "ao_integrals" tags=[:system, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx, overlap, kinetic, nuclear
using ElemCo.IntegralTools: generate_ao_fdump, ao_to_mo!
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
    fd_mo = ao_to_mo!(deepcopy(fd), X)

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

    # rectangular transform: keep only the first m MOs (drop the rest, e.g. deleted orbitals)
    m = nao - 2
    fd_red = ao_to_mo!(deepcopy(fd), X[:, 1:m])
    @test size(fd_red.int2, 1) == m
    @test headvar(fd_red, "NORB", Int) == m
    @test length(fd_red.head["ORBSYM"]) == m
    vred = detri_int2(fd_red.int2, m, 1:m, 1:m, 1:m, 1:m)
    @test maximum(abs.(vred .- vmo_ref[1:m, 1:m, 1:m, 1:m])) < 1e-10
    @test maximum(abs.(fd_red.int1 .- X[:, 1:m]' * hAO * X[:, 1:m])) < 1e-10
  end

  @testset "exact HF (AO integrals) vs reference RHF" begin
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
    res = ElemCo.hf(EC2)
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

@testset "HF/UHF on AO integrals (@ints / @hf / @uhf)" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  basis = Dict("ao"=>"sto-3g")
  Eref = -74.96485912107553

  # independent reference UHF from the dense chemists' tensor G
  function ref_uhf(S, h, G, Enuc, na, nb; maxit=300, thr=1e-12)
    nb_ = size(S, 1)
    Fe = eigen(Symmetric(S)); X = Fe.vectors * Diagonal(1.0 ./ sqrt.(Fe.values)) * Fe.vectors'
    _, C = eigen(Symmetric(X' * h * X)); Ca = X * C; Cb = X * C
    E = 0.0
    for _ in 1:maxit
      Da = Ca[:, 1:na] * Ca[:, 1:na]'
      Db = Cb[:, 1:nb] * Cb[:, 1:nb]'
      Dt = Da + Db
      J = zeros(nb_, nb_); Ka = zeros(nb_, nb_); Kb = zeros(nb_, nb_)
      @inbounds for q in 1:nb_, p in 1:nb_
        jpq = 0.0; kapq = 0.0; kbpq = 0.0
        for s in 1:nb_, r in 1:nb_
          jpq += G[p, q, r, s] * Dt[r, s]    # (pq|rs)
          kapq += G[p, s, r, q] * Da[r, s]   # (ps|rq)
          kbpq += G[p, s, r, q] * Db[r, s]
        end
        J[p, q] = jpq; Ka[p, q] = kapq; Kb[p, q] = kbpq
      end
      Fa = h + J - Ka; Fb = h + J - Kb
      Enew = 0.5 * sum(Da .* (h + Fa)) + 0.5 * sum(Db .* (h + Fb)) + Enuc
      _, Cc = eigen(Symmetric(X' * Fa * X)); Ca = X * Cc
      _, Cd = eigen(Symmetric(X' * Fb * X)); Cb = X * Cd
      if abs(Enew - E) < thr; E = Enew; break; end
      E = Enew
    end
    return E
  end

  # @hf auto-generates AO integrals when EC.fd is empty
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @test isempty(EC.fd)
  e_hf = @hf
  @test is_ao_basis(EC.fd)
  @test abs(e_hf["HF"] - Eref) < 1e-7

  # explicit @ints, then @hf reuses the stored integrals
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @ints
  @test is_ao_basis(EC.fd)
  e_hf2 = @hf
  @test abs(e_hf2["HF"] - Eref) < 1e-7

  # closed-shell UHF must reduce to RHF
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  e_uhf = @uhf
  @test abs(e_uhf["UHF"] - Eref) < 1e-7

  # open-shell UHF (water cation, ms2=1) vs independent reference UHF
  bao = generate_basis(parse_geometry(geometry, basis), "ao")
  G = eri_2e4idx(bao); S = overlap(bao); hAO = kinetic(bao) + nuclear(bao)
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @set wf charge=1 ms2=1
  e_uhf_cation = @uhf
  Enuc = EC.fd.int0
  ref = ref_uhf(S, hAO, G, Enuc, 5, 4)   # 9 e⁻, ms2=1 → nα=5, nβ=4
  @test abs(e_uhf_cation["UHF"] - ref) < 1e-6

  # guard: open-shell AO integrals are not yet supported for correlated methods
  # (EC still holds the open-shell water cation built by the @uhf above)
  @test_throws ErrorException (@cc "ccsd")

  # closed-shell: AO-direct CCSD/DCSD and the auto AO→MO switch (CCSD(T), FCI) must all
  # agree with the explicit AO→MO transform reference (ao_to_mo!).
  let load_orbitals = ElemCo.OrbTools.load_orbitals,
      ao_to_mo! = ElemCo.IntegralTools.ao_to_mo!
    # AO-direct CCSD/DCSD: EC.fd stays in the AO basis
    for m in ("ccsd", "dcsd")
      key = uppercase(m)
      EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
      @hf
      ao_fd = EC.fd
      cMO = Matrix(load_orbitals(EC).α)
      e_ao = @cc m
      @test is_ao_basis(EC.fd)
      EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
      EC.fd = ao_to_mo!(deepcopy(ao_fd), cMO)
      e_ref = @cc m
      @test abs(e_ao["HF"]  - e_ref["HF"])  < 1e-9
      @test abs(e_ao[key]   - e_ref[key])   < 1e-6
    end
    nao_ref = size(S, 1)   # AOs in this sto-3g water (== orbitals when nothing is dropped)
    # each EC gets its own orbital dump in its (unique) scratch dir, so differently-sized
    # systems below don't clash over the shared default "wf.h5" in the working directory
    fresh(b=basis) = (e = ElemCo.ECInfo(system=parse_geometry(geometry, b));
                      e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)

    # all-electron FCI: the auto AO→MO switch matches FCI on the explicit MO dump (pure
    # basis switch — no orbitals dropped, so it isolates the transform from any folding)
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    ref_fd = ao_to_mo!(deepcopy(EC.fd), cMO)        # snapshot AO→MO before the in-place switch
    e_ao_fci = @fci                                 # @fci switches EC.fd to the MO basis in place
    @test !is_ao_basis(EC.fd)
    @test headvar(EC.fd, "NORB", Int) == nao_ref    # nothing dropped
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    EC.fd = ref_fd
    e_ref_fci = @fci
    @test abs(e_ao_fci["FCI"] - e_ref_fci["FCI"]) < 1e-7

    # frozen-core folding: the auto switch folds the core out of the MO dump (NORB shrinks);
    # the result must match an explicit fold (freeze_orbs_in_dump) of the same orbital.
    EC = fresh()
    EC.options.wf.freeze_nocc = 1
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    fold_ref = ao_to_mo!(deepcopy(EC.fd), cMO)
    e_fold = @cc "ccsd(t)"                           # auto switch folds 1 core orbital into the dump
    @test headvar(EC.fd, "NORB", Int) == nao_ref - 1
    EC = fresh()
    EC.fd = fold_ref
    ElemCo.DumpTools.freeze_orbs_in_dump(EC, [1])    # explicit fold of the lowest orbital
    EC.options.wf.freeze_nocc = 0                    # already folded — don't freeze again
    e_ref = @cc "ccsd(t)"
    @test headvar(EC.fd, "NORB", Int) == nao_ref - 1
    @test abs(e_fold["HF"]      - e_ref["HF"])      < 1e-9
    @test abs(e_fold["CCSD(T)"] - e_ref["CCSD(T)"]) < 1e-8

    # redundant (linearly-dependent) orbitals: a high redthr forces an orbital to be deleted.
    # The auto switch drops it from the transform (NORB shrinks); FCI must match the explicit
    # transform with the same orbital dropped. (ccsd/dcsd stay AO-direct and don't drop yet.)
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    EC.options.scf.redthr = 0.4                     # delete the most linearly-dependent orbital
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    ndel = ElemCo.OrbTools.n_deleted_orbitals(EC)
    @test ndel == 1
    red_ref = ao_to_mo!(deepcopy(EC.fd), cMO[:, 1:nao_ref-ndel])   # explicit drop of the same orbital
    e_ao = @fci                                     # auto switch drops the deleted orbital
    @test headvar(EC.fd, "NORB", Int) == nao_ref - ndel
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    EC.fd = red_ref
    e_ref = @fci
    @test abs(e_ao["FCI"] - e_ref["FCI"]) < 1e-7
  end
end
end
