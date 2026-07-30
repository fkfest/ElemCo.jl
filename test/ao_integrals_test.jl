@testitem "ao_integrals" tags=[:system, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.MSystems: parse_geometry, nuclear_repulsion
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: eri_2e4idx, overlap, kinetic, nuclear
using ElemCo.IntegralTools: ao_integrals, generate_mo_dump
using ElemCo.PMStore: pm_exists, pm_to_joint!
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.TensorTools: detri_int2, load2idx, mmap3idx, @mtensor, @tensor
using LinearAlgebra

@testset "AO integral files (non-DF AO integrals)" begin
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

  Enuc = ao_integrals(EC)      # writes the ± supermatrix store + "S_AA"/"h_AA", returns Enuc

  @testset "files and metadata" begin
    @test isempty(EC.fd)                        # EC.fd is MO-only, never holds AO integrals
    @test pm_exists(EC)                         # the ± supermatrix store is the AO representation
    @test !file_exists(EC, "ao_int2")           # no jointly packed AO integrals are ever written
    @test abs(Enuc - nuclear_repulsion(EC.system)) < 1e-12
    @test maximum(abs.(load2idx(EC, "S_AA") .- S)) < 1e-12
    @test maximum(abs.(load2idx(EC, "h_AA") .- hAO)) < 1e-12
  end

  pm_to_joint!(EC)      # reconstruct the jointly packed integrals from the store for the round-trip
  aofile, aoint2 = mmap3idx(EC, "ao_int2")
  @testset "physicist-notation round-trip" begin
    # detri reconstructs full <pq|rs> from the triangular storage;
    # it must equal (pr|qs) = G[p,r,q,s]
    sp = 1:nao
    v = detri_int2(aoint2, nao, sp, sp, sp, sp)    # v[p,q,r,s] = <pq|rs>
    Gphys = permutedims(G, (1,3,2,4))              # G[p,r,q,s] -> [p,q,r,s]
    @test maximum(abs.(v .- Gphys)) < 1e-12
  end
  close(aofile)

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
    fd_mo = generate_mo_dump(EC, X)          # builds the MO dump in EC.fd from the AO files
    @test EC.fd === fd_mo
    @test !isempty(EC.fd)

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
    @test headvar(fd_mo, "NORB", Int) == nao
    @test abs(fd_mo.int0 - Enuc) < 1e-12

    # rectangular transform: keep only the first m MOs (drop the rest, e.g. deleted orbitals)
    m = nao - 2
    fd_red = generate_mo_dump(EC, X[:, 1:m])
    @test size(fd_red.int2, 1) == m
    @test headvar(fd_red, "NORB", Int) == m
    @test length(fd_red.head["ORBSYM"]) == m
    vred = detri_int2(fd_red.int2, m, 1:m, 1:m, 1:m, 1:m)
    @test maximum(abs.(vred .- vmo_ref[1:m, 1:m, 1:m, 1:m])) < 1e-10
    @test maximum(abs.(fd_red.int1 .- X[:, 1:m]' * hAO * X[:, 1:m])) < 1e-10
    EC.fd = FDump{Float64,3}()   # clean up for the following tests
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

    Eref = ref_rhf(S, hAO, G, Enuc, 5)   # water: 10 e⁻ → 5 doubly occupied

    EC2 = ECInfo{Float64}()
    EC2.system = parse_geometry(geometry, Dict("ao"=>"sto-3g"))
    res = ElemCo.hf(EC2)                 # generates the AO integral files itself
    @test abs(res["HF"] - Eref) < 1e-7
    @test isempty(EC2.fd)                # HF does not touch EC.fd
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

  # @hf auto-generates the AO integral files when they are missing
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @test isempty(EC.fd)
  e_hf = @hf
  @test isempty(EC.fd)                       # EC.fd is MO-only; AO integrals live on files
  @test pm_exists(EC)
  @test abs(e_hf["HF"] - Eref) < 1e-7

  # explicit @ints, then @hf reuses the stored integrals
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @ints
  @test pm_exists(EC)
  e_hf2 = @hf
  @test abs(e_hf2["HF"] - Eref) < 1e-7

  # @dummy changes the nuclear charges → the 1-e core Hamiltonian (h_AA) must be invalidated,
  # but the 2-e AO integrals are unchanged (ghost atoms keep their basis functions)
  # and must be KEPT (they are the expensive part).
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, basis))
  @ints
  @test pm_exists(EC)
  @test file_exists(EC, "h_AA")
  @dummy ["O"]
  @test pm_exists(EC)                 # 2-e integrals kept (dummy-independent)
  @test !file_exists(EC, "h_AA")      # 1-e core Hamiltonian invalidated -> recomputed on demand

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
  Enuc = nuclear_repulsion(EC.system)
  ref = ref_uhf(S, hAO, G, Enuc, 5, 4)   # 9 e⁻, ms2=1 → nα=5, nβ=4
  @test abs(e_uhf_cation["UHF"] - ref) < 1e-6

  # closed-shell: AO-direct CCSD/DCSD and the automatic AO→MO derivation (CCSD(T), FCI)
  # must all agree with an explicitly generated MO dump (generate_mo_dump).
  let load_orbitals = ElemCo.OrbTools.load_orbitals
    nao_ref = size(S, 1)   # AOs in this sto-3g water (== orbitals when nothing is dropped)
    # each EC gets its own orbital dump in its (unique) scratch dir, so differently-sized
    # systems below don't clash over the shared default "wf.h5" in the working directory
    fresh(b=basis) = (e = ElemCo.ECInfo(system=parse_geometry(geometry, b));
                      e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)

    # OPEN-SHELL AO→MO derivation: correlated methods on an AO-source open-shell reference derive
    # an unrestricted (UHF) MO dump. Validation: a *closed-shell* molecule run via @uhf must
    # reproduce the restricted CCSD (exercises the full unrestricted transform + frozen-core fold
    # + UCCSD path against the restricted result).
    EC = fresh(); @uhf; e_ucc = @cc ccsd
    EC = fresh(); @hf;  e_rcc = @cc ccsd
    @test abs(e_ucc["UCCSD"] - e_rcc["CCSD"]) < 1e-7
    # open-shell water cation (ms2=1): @uhf then correlated methods run on the derived UHF dump.
    EC = fresh(); @set wf charge=1 ms2=1
    e_cat_uhf = @uhf
    e_cat = @cc ccsd
    @test abs(e_cat["HF"] - e_cat_uhf["UHF"]) < 1e-8   # reference HF energy consistent
    @test isempty(EC.fd)                               # transient UHF dump discarded
    e_cat2 = @cc ccsd                                  # deterministic across repeated calls
    @test abs(e_cat["UCCSD"] - e_cat2["UCCSD"]) < 1e-10
    e_cat_dcsd = @cc dcsd
    @test isfinite(e_cat_dcsd["UDCSD"]) && e_cat_dcsd["UDCSD"] < e_cat["HF"]
    @test !ElemCo.OrbTools.is_restricted(load_orbitals(EC))   # genuinely unrestricted reference
    @test isempty(EC.fd)                                      # AO-direct left EC.fd empty

    # independent check of the unrestricted transform's αβ block for the (genuinely cα≠cβ) cation:
    # int2ab[p,q,r,s] = <pq|rs> (p,r α; q,s β) = (pr|qs) = Σ G[μρ|νσ] cα[μ,p] cα[ρ,r] cβ[ν,q] cβ[σ,s]
    reduce_idx(A, X, d) = (perm = (d, setdiff(1:4, d)...);
        Ap = permutedims(A, perm); n = size(Ap, 1);
        Bp = reshape(X' * reshape(Ap, n, :), size(X, 2), size(Ap, 2), size(Ap, 3), size(Ap, 4));
        permutedims(Bp, invperm(collect(perm))))
    ca = Matrix(load_orbitals(EC).α); cb = Matrix(load_orbitals(EC).β)
    @test maximum(abs.(ca .- cb)) > 0.1                     # genuinely open-shell orbitals
    ECab = fresh(); ao_integrals(ECab); generate_mo_dump(ECab, ElemCo.QMTensors.SpinMatrix(copy(ca), copy(cb)))
    ref_ab = permutedims(reduce_idx(reduce_idx(reduce_idx(reduce_idx(G, ca, 1), ca, 2), cb, 3), cb, 4), (1,3,2,4))
    @test maximum(abs.(ECab.fd.int2ab .- ref_ab)) < 1e-10

    # AO-DIRECT open-shell UCCSD/UDCSD/UMP2 (cation, ms2=1): the @uhf → @cc runs above went
    # AO-direct (EC.fd empty). Their energies must match a derived-UHF-MO-dump reference built from
    # the SAME deterministic UHF orbitals (int.ao_direct=false forces the derive route, with the
    # same default frozen core) — this validates the occ-early unrestricted dressing + AO kext.
    EC = fresh(); @set wf charge=1 ms2=1
    EC.options.int.ao_direct = false
    e_cat_mo_uhf = @uhf
    e_cat_mo   = @cc ccsd                                      # derived UHF MO dump (not AO-direct)
    e_cat_mo_d = @cc dcsd
    @test abs(e_cat["HF"]         - e_cat_mo["HF"])      < 1e-8   # same UHF reference
    @test abs(e_cat["UMP2"]       - e_cat_mo["UMP2"])    < 1e-8   # AO-direct UMP2  == derive UMP2
    @test abs(e_cat["UCCSD"]      - e_cat_mo["UCCSD"])   < 1e-7   # AO-direct UCCSD == derive UCCSD
    @test abs(e_cat_dcsd["UDCSD"] - e_cat_mo_d["UDCSD"]) < 1e-7   # AO-direct UDCSD == derive UDCSD

    # AO-direct CCSD/DCSD (no frozen/deleted orbitals): EC.fd stays empty (integrals read from
    # the AO files) and the energy matches an all-electron MO dump from the same orbitals.
    for m in ("ccsd", "dcsd")
      key = uppercase(m)
      EC = fresh()
      EC.options.wf.freeze_nocc = 0            # keep it all-electron so the run stays AO-direct
      @hf
      cMO = Matrix(load_orbitals(EC).α)
      e_ao = @cc m
      @test isempty(EC.fd)                     # nothing parked in EC.fd by the AO-direct run
      # reference: explicit all-electron MO dump from the same orbitals (external dumps not folded)
      EC = fresh()
      ao_integrals(EC)
      generate_mo_dump(EC, cMO)
      e_ref = @cc m
      @test abs(e_ao["HF"]  - e_ref["HF"])  < 1e-9
      @test abs(e_ao[key]   - e_ref[key])   < 1e-6
      # AO-direct now also computes MP2 (from the bare d_oovv) and uses it as the CC start guess;
      # it must match the MO-dump MP2 exactly
      @test abs(e_ao["MP2"] - e_ref["MP2"]) < 1e-9
    end

    # STANDALONE AO-direct MP2 / UMP2 (`@cc mp2`, not as a CC start guess): the method gate now admits
    # (U/R)MP2, so a bare `@cc mp2` on AO files runs AO-direct (EC.fd stays empty) off the bare
    # d_oovv/d_OOVV/d_oOvV blocks + Fock, instead of deriving a full MO dump. Energies must match the
    # derived-MO-dump MP2 from the same orbitals (int.ao_direct=false forces the derive route).
    EC = fresh(); @hf; e_smp2 = @cc mp2                       # closed-shell, default frozen core
    @test isempty(EC.fd)                                      # standalone MP2 stayed AO-direct
    EC = fresh(); EC.options.int.ao_direct = false; @hf; e_smp2_ref = @cc mp2   # derive-route reference
    @test abs(e_smp2["MP2"] - e_smp2_ref["MP2"]) < 1e-9
    # open-shell standalone UMP2 (water cation): `@cc mp2` on UHF orbitals dispatches to UMP2
    EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_sump2 = @cc mp2
    @test isempty(EC.fd)                                      # AO-direct unrestricted MP2
    EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_sump2_ref = @cc mp2
    @test abs(e_sump2["UMP2"] - e_sump2_ref["UMP2"]) < 1e-8

    # The T1-dressed AO Fock is NON-Hermitian (bra uses C̃ᴸ, ket C̃ᴿ), so f_vo ≠ f_ovᵀ: ao_dressed_ints
    # must build the v,o Fock block from d_vooo, not by transposing f_ov. That block is latent on the
    # kext-only AO-direct energy path (R1 is seeded from dh_mm[v,o]), so the CCSD-energy checks above
    # cannot catch a wrong f_vo — compare df_mm directly to a dense dressed-Fock reference at nonzero T1.
    EC = fresh(); EC.options.wf.freeze_nocc = 0; @hf
    cMO = Matrix(load_orbitals(EC).α); nao_d = size(cMO, 1)
    ElemCo.CoupledCluster.ao_cc_setup!(EC; closed_shell=true)   # closed-shell residual on RHF orbitals
    occ = EC.space['o']; virt = EC.space['v']; no_d = length(occ); nv_d = length(virt)
    T1d = 0.03 .* reshape(range(-1.0, 1.0; length=nv_d*no_d), nv_d, no_d)   # deterministic nonzero singles
    ElemCo.CoupledCluster.ao_dressed_ints(EC, T1d, cMO)
    df = load2idx(EC, "df_mm")
    pm_to_joint!(EC)                                                        # dense reference from the store
    aofile, int2m = mmap3idx(EC, "ao_int2")
    vAO = detri_int2(int2m, nao_d, 1:nao_d, 1:nao_d, 1:nao_d, 1:nao_d)      # dense ⟨μν|ρσ⟩
    close(aofile)
    heff = load2idx(EC, "h1eff_AA")
    CL = copy(cMO); CR = copy(cMO)
    CL[:, virt] .-= cMO[:, occ] * T1d'                                      # C̃ᴸ = [C_o | C_v − C_o·T1ᵀ]
    CR[:, occ]  .+= cMO[:, virt] * T1d                                      # C̃ᴿ = [C_o + C_v·T1 | C_v]
    Docc = CL[:, occ] * CR[:, occ]'                                         # dressed occupied density
    dref = CL' * heff * CR                                                  # dressed h̃
    @mtensor J[p,q] := vAO[μ,ν,ρ,σ] * CL[μ,p] * CR[ρ,q] * Docc[ν,σ]
    @mtensor K[p,q] := vAO[μ,ν,ρ,σ] * CL[μ,p] * CR[σ,q] * Docc[ν,ρ]
    dref .+= 2.0 .* J .- K                                                  # + Σ_k (2⟨pk|qk⟩ − ⟨pk|kq⟩)
    @test maximum(abs.(df .- dref)) < 1e-9                                  # AO-direct dressed Fock == exact (all blocks incl. v,o)
    @test maximum(abs.(df[virt,occ] .- permutedims(df[occ,virt],(2,1)))) > 1e-2   # genuinely non-Hermitian (f_vo ≠ f_ovᵀ)

    # AO-direct frozen core: with the default core freezing (wf.core=:auto) a closed-shell @cc ccsd
    # from the AO files runs AO-DIRECT (EC.fd stays empty) but folds the O 1s core into an effective
    # 1-e Hamiltonian — the energy must match an explicit MO-dump fold (freeze_orbs_in_dump) and
    # differ from the all-electron result.
    EC = fresh()
    @hf                                        # default wf.core = :auto -> 1 core orbital frozen
    cMO = Matrix(load_orbitals(EC).α)
    e_fc = @cc ccsd                            # AO-direct with frozen-core folding
    @test isempty(EC.fd)                       # AO-direct: nothing parked in EC.fd
    EC = fresh()
    ao_integrals(EC)
    generate_mo_dump(EC, cMO)
    ElemCo.DumpTools.freeze_orbs_in_dump(EC, [1])   # explicit fold of the lowest orbital
    EC.options.wf.freeze_nocc = 0
    e_fc_ref = @cc ccsd
    @test abs(e_fc["HF"]   - e_fc_ref["HF"])   < 1e-9
    @test abs(e_fc["CCSD"] - e_fc_ref["CCSD"]) < 1e-8
    @test abs(e_fc["MP2"]  - e_fc_ref["MP2"])  < 1e-9   # frozen-core AO-direct MP2 == folded-dump MP2
    # all-electron reference (unfolded dump) must differ — confirms the core was actually frozen
    EC = fresh()
    ao_integrals(EC)
    generate_mo_dump(EC, cMO)
    e_ae_ref = @cc ccsd
    @test abs(e_fc["CCSD"] - e_ae_ref["CCSD"]) > 1e-5   # frozen-core ≠ all-electron

    # int.ao_direct=false forces the derived-MO-dump route for closed-shell CCSD; the (folded)
    # frozen-core energy must equal the AO-direct one.
    EC = fresh()
    EC.options.int.ao_direct = false
    @hf
    e_mo = @cc ccsd                            # derived MO dump (folded), not AO-direct
    @test isempty(EC.fd)                       # transient dump discarded
    @test abs(e_mo["CCSD"] - e_fc["CCSD"]) < 1e-8   # same frozen-core CCSD as AO-direct

    # all-electron FCI: the automatic AO→MO derivation matches FCI on the explicit MO dump
    # (pure basis change — no orbitals dropped, so it isolates the transform from any folding)
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    e_ao_fci = @fci                                 # derives a transient MO dump internally
    @test isempty(EC.fd)                            # ... and discards it afterwards
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    ao_integrals(EC)
    generate_mo_dump(EC, cMO)
    e_ref_fci = @fci
    @test abs(e_ao_fci["FCI"] - e_ref_fci["FCI"]) < 1e-7

    # frozen-core folding: the automatic derivation folds the core out of the transient MO
    # dump; the result must match an explicit fold (freeze_orbs_in_dump) of the same orbital —
    # and the user's freeze options must not be modified by the run.
    EC = fresh()
    EC.options.wf.freeze_nocc = 1
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    e_fold = @cc "ccsd(t)"                           # derivation folds 1 core orbital
    @test EC.options.wf.freeze_nocc == 1             # options are not mutated by the driver
    @test EC.options.wf.core == :auto
    EC = fresh()
    ao_integrals(EC)
    generate_mo_dump(EC, cMO)
    ElemCo.DumpTools.freeze_orbs_in_dump(EC, [1])    # explicit fold of the lowest orbital
    @test headvar(EC.fd, "NORB", Int) == nao_ref - 1
    EC.options.wf.freeze_nocc = 0                    # already folded — don't freeze again
    e_ref = @cc "ccsd(t)"
    @test abs(e_fold["HF"]      - e_ref["HF"])      < 1e-9
    @test abs(e_fold["CCSD(T)"] - e_ref["CCSD(T)"]) < 1e-8

    # rerunning with a different freeze setting is honored (the transient dump is re-derived):
    # freeze_nocc=0 after the frozen-core run above must give the all-electron CCSD(T).
    EC = fresh()
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    EC.options.wf.freeze_nocc = 1
    e_fc = @cc "ccsd(t)"
    EC.options.wf.freeze_nocc = 0
    e_ae = @cc "ccsd(t)"
    @test abs(e_fc["CCSD(T)"] - e_fold["CCSD(T)"]) < 1e-8
    @test abs(e_ae["CCSD(T)"] - e_fc["CCSD(T)"]) > 1e-5   # all-electron ≠ frozen-core
    # ... and an AO-direct method still works after a derived-dump run
    e_ao2 = @cc ccsd
    @test isempty(EC.fd)

    # @moints: the user-facing AO→MO transform. Unlike the dump a driver derives for itself it
    # PERSISTS (the mmapped MO integrals survive `delete_temporary_files!` at the end of a run), so
    # it can be written out and reused. It must reproduce the AO-direct energies exactly, both
    # closed- and open-shell, and a re-read of the written FCIDUMP must reproduce them too.
    EC = fresh()
    @hf                                              # default frozen core (1 core orbital)
    e_ao = @cc ccsd
    @test isempty(EC.fd)                             # AO-direct
    EC = fresh()
    @hf
    @moints
    @test !isempty(EC.fd)                            # MO integrals parked in EC.fd ...
    @test headvar(EC.fd, "NORB", Int) == nao_ref - 1 #  ... reduced to the active space (core folded)
    fcidump_file = joinpath(EC.scr, "MOINTS_FCIDUMP")
    @write_ints fcidump_file
    e_mo1 = @cc ccsd
    e_mo2 = @cc ccsd                                 # the dump survives a driver run and is reused
    @test !isempty(EC.fd)
    @test abs(e_mo1["CCSD"] - e_ao["CCSD"]) < 1e-8
    @test abs(e_mo2["CCSD"] - e_mo1["CCSD"]) < 1e-12
    EC = fresh(); EC.fd = read_fcidump(fcidump_file) # the written FCIDUMP is a complete input
    EC.options.wf.freeze_nocc = 0                    # already folded
    e_fd = @cc ccsd
    @test abs(e_fd["HF"]   - e_ao["HF"])   < 1e-9
    @test abs(e_fd["CCSD"] - e_ao["CCSD"]) < 1e-8
    # open shell: @moints builds the unrestricted (UHF) dump
    EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_cat_ao = @cc ccsd
    EC = fresh(); @set wf charge=1 ms2=1; @uhf; @moints
    @test EC.fd.uhf
    e_cat_mo = @cc ccsd
    @test abs(e_cat_mo["UCCSD"] - e_cat_ao["UCCSD"]) < 1e-8
    # ... and the FCIDUMP written from a CHARGED calculation is self-contained: its NELEC is the
    # cation's electron count (the writer applies `wf.charge`), so reading it back needs no charge
    cat_file = joinpath(EC.scr, "MOINTS_CATION_FCIDUMP")
    @write_ints cat_file
    fd_cat = read_fcidump(cat_file)
    @test headvar(fd_cat, "NELEC", Int) == 10 - 1 - 2   # neutral 10, cation, 1 folded core orbital
    @test headvar(fd_cat, "MS2", Int) == 1
    EC = fresh(); EC.fd = fd_cat                        # no charge/ms2 set — the dump says it all
    EC.options.wf.freeze_nocc = 0
    e_cat_fd = @cc ccsd
    @test abs(e_cat_fd["HF"]    - e_cat_ao["HF"])    < 1e-9
    @test abs(e_cat_fd["UCCSD"] - e_cat_ao["UCCSD"]) < 1e-8
    # @moints without AO integrals on file generates them itself (like @ints)
    EC = fresh(); @hf; ElemCo.IntegralTools.delete_ao_integrals!(EC)
    @test !pm_exists(EC)
    @moints
    @test pm_exists(EC) && !isempty(EC.fd)

    # redundant (linearly-dependent) orbitals: a high redthr forces an orbital to be deleted.
    # The automatic derivation drops it from the transform; FCI must match the explicit
    # transform with the same orbital dropped.
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    EC.options.scf.redthr = 0.4                     # delete the most linearly-dependent orbital
    @hf
    cMO = Matrix(load_orbitals(EC).α)
    ndel = ElemCo.OrbTools.n_deleted_orbitals(EC)
    @test ndel == 1
    e_ao = @fci                                     # derivation drops the deleted orbital
    EC = fresh()
    EC.options.wf.freeze_nocc = 0
    ao_integrals(EC)
    generate_mo_dump(EC, cMO[:, 1:nao_ref-ndel])    # explicit drop of the same orbital
    e_ref = @fci
    @test abs(e_ao["FCI"] - e_ref["FCI"]) < 1e-7
  end
end
end
