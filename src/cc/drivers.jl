"""
    Drivers

Module for methods drivers (ccdriver, dfccdriver, etc).
"""
module Drivers
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.TensorTools
using ..ElemCo.IntegralTools
using ..ElemCo.CCTools
using ..ElemCo.CoupledCluster
using ..ElemCo.DMRG
using ..ElemCo.DFCoupledCluster
using ..ElemCo.FciDumps
using ..ElemCo.DumpTools
using ..ElemCo.OrbTools
using ..ElemCo.Properties
using ..ElemCo.FockFactory
using ..ElemCo.PMStore
using ..ElemCo.FCI
using ..ElemCo.EOM
using ..ElemCo.DfDump: dfdump
using LinearAlgebra: diag

export ccdriver, dfccdriver, fcidriver, extrapolate
export mo_integrals

"""
    need_correlated_properties(EC::ECInfo)

  Return `true` when the conventional or DF correlated drivers must construct a
  correlated 1-RDM for property output or natural-orbital storage.
"""
need_correlated_properties(EC::ECInfo) = EC.options.cc.properties || !isempty(EC.options.wf.natorb)

"""
    need_lagrange_multipliers(EC::ECInfo, method::ECMethod)

  Return `true` when the calculation requires lambda amplitudes either because
  the requested method is a Λ-variant or because post-processing needs the
  correlated density.
"""
need_lagrange_multipliers(EC::ECInfo, method::ECMethod) = has_prefix(method, "Λ") || need_correlated_properties(EC)

"""
    need_fci_properties(EC::ECInfo, ciphi::Bool)

  Return `true` when FCI or CIPHI must construct correlated property data for
  dipole output or natural-orbital storage.
"""
need_fci_properties(EC::ECInfo, ciphi::Bool) = (ciphi ? EC.options.ciphi.properties : EC.options.fci.properties) || !isempty(EC.options.wf.natorb)

"""
    dfmp2_property_rdm(EC::ECInfo, method::ECMethod)

  Build the restricted DF-MP2 correlated 1-RDM directly from the saved doubles
  amplitudes. DF-MP2 properties use `U2 = T2` and do not require a separate
  Lagrange-multiplier solve.
"""
function dfmp2_property_rdm(EC::ECInfo{T}, method::ECMethod) where T
  if is_unrestricted(method) || has_prefix(method, "R")
    error("DF-MP2 correlated properties are only implemented for restricted methods.")
  end
  empty_singles = zeros(T, 0, 0)
  T2 = read_starting_guess4amplitudes(EC, Val(2))
  return CoupledCluster.calc_correlated_1rdm(EC, method, empty_singles, T2, empty_singles, T2)
end

"""
    write_fci_dipole(EC::ECInfo, ciphi::Bool)

  Return `true` when the FCI or CIPHI driver should print and store dipole
  components in the returned output dictionary.
"""
write_fci_dipole(EC::ECInfo, ciphi::Bool) = ciphi ? EC.options.ciphi.properties : EC.options.fci.properties

"""
    has_perturbative_triples(method::ECMethod)

  Return `true` when `method` carries a perturbative triples correction.
"""
has_perturbative_triples(method::ECMethod) = method.exclevel[3] ∈ [:pert, :pertiter]

"""
    defer_lambda_perturbative_triples(method::ECMethod)

  Return `true` when perturbative triples must be evaluated after the lambda
  equations rather than in the initial energy-only pass.
"""
defer_lambda_perturbative_triples(method::ECMethod) = has_prefix(method, "Λ") && has_perturbative_triples(method)

function add_perturbative_triples(EC::ECInfo, ecmethod::ECMethod, energies::OutDict;
                                  save_pert_t3::Bool=false, timer=time_ns())
  if is_similarity_transformed(EC.fd) && !has_prefix(ecmethod, "Λ")
    warnerror("Perturbative triples for similarity transformed Hamiltonians must be calculated
      with ΛCCSD(T) method! The error can be ignored by setting the option `cc.ignore_error=true`.",
      !EC.options.cc.ignore_error)
  end
  main_name = method_name(ecmethod)
  ECC = energies[main_name*"c"]
  EHF = energies["HF"]
  ET3, ET3b = values(calc_pertT(EC, ecmethod; save_t3=save_pert_t3))
  println()
  output_E_method(ECC+ET3b+EHF, main_name*"[T]", "total energy:      ")
  output_E_method(ECC+ET3, main_name*"(T)", "correlation energy:")
  output_E_method(ECC+ET3+EHF, main_name*"(T)", "total energy:       ")
  println()
  timer = print_time(EC, timer, "(T)", 1)
  energies_out = copy(energies)
  push!(energies_out, "[T]"=>(ET3b,"[T] energy contribution"),
                      "(T)"=>(ET3,"(T) energy contribution"),
                      main_name*"(T)c"=>(ECC+ET3,"$main_name(T) correlation energy"),
                      main_name*"(T)"=>(ECC+ET3+EHF,"$main_name(T) total energy"))
  return energies_out, timer
end

"""
    correlated_dipole(EC::ECInfo, rdm::SpinMatrix)

  Evaluate the dipole moment from a correlated MO-space 1-RDM when the current
  wavefunction dump still contains orbitals and AO-basis metadata.
"""
function correlated_dipole(EC::ECInfo, rdm::SpinMatrix)
  if !has_dumpfile(EC) || isempty(EC.system)
    return nothing
  end
  orbital_data = fetch_orbital_data(EC)
  if isnothing(orbital_data) || isempty(orbital_data.basis)
    return nothing
  end
  return calc_dipole_moment(EC, orbital_data.cMO, rdm; basis=orbital_data.basis)
end

"""
    full_space_fci_rdm(EC::ECInfo, rdm_active::AbstractMatrix, occ_key::Char, occval)

  Embed an active-space FCI/CIPHI 1-RDM into the full orbital space and restore
  the frozen occupied reference contribution.
"""
function full_space_fci_rdm(EC::ECInfo, rdm_active::AbstractMatrix, occ_key::Char, occval)
  space_save, space_b4freeze = restore_full_space!(EC)
  frozen_space = save_space(EC)
  rdm_sym = 0.5 * (rdm_active + rdm_active')
  full_size = length(space_b4freeze[':'])
  if size(rdm_sym, 1) == full_size
    rdm = Matrix(rdm_sym)
  else
    active_orbs = sort(union(frozen_space['o'], frozen_space['v'], frozen_space['O'], frozen_space['V']))
    @assert size(rdm_sym, 1) == length(active_orbs)
    rdm = zeros(eltype(rdm_sym), full_size, full_size)
    rdm[active_orbs, active_orbs] = rdm_sym
  end
  frozen_occs = setdiff(space_b4freeze[occ_key], frozen_space[occ_key])
  CoupledCluster.add_reference_density!(rdm, frozen_occs, occval)
  restore_space!(EC, space_save)
  return rdm
end

"""
    fci_property_rdms(EC::ECInfo, rdm_a::AbstractMatrix, rdm_b::AbstractMatrix, closed_shell::Bool)

  Construct the dipole and storage 1-RDM payloads used by FCI and CIPHI property
  post-processing in the full orbital space.
"""
function fci_property_rdms(EC::ECInfo, rdm_a::AbstractMatrix, rdm_b::AbstractMatrix, closed_shell::Bool)
  occa_key = 'o'
  occb_key = closed_shell ? 'o' : 'O'
  occval = one(eltype(rdm_a))
  rdma_full = full_space_fci_rdm(EC, rdm_a, occa_key, occval)
  rdmb_full = full_space_fci_rdm(EC, rdm_b, occb_key, occval)
  dipole_rdm = SpinMatrix(rdma_full, rdmb_full)
  storage_rdm = closed_shell ? SpinMatrix(copy(rdma_full + rdmb_full)) : copy(dipole_rdm)
  return dipole_rdm, storage_rdm
end

"""
    ao_direct_method(ecm::ECMethod) -> Bool

  Whether `ecm` runs directly on the AO integral files without a preceding MO transform:
  MP2/UMP2/RMP2, or CCSD/DCSD/CCD/DCD optionally with perturbative triples and/or a Λ, EOM, QV,
  orbital-optimizing (`O`) or Brueckner (`B`) prefix. FCI and iterative triples always derive a
  transient MO dump (see [`derive_mo_basis!`](@ref)).
"""
function ao_direct_method(ecm::ECMethod)
  name = uppercase(method_name(ecm; root=true))         # base method (EOM/U/R/Λ/QV prefixes stripped)
  name == "MP2" && return true                          # standalone MP2/UMP2/RMP2 off the bare AO blocks
  # SVD triples (SVD-DC-CCSDT / SVD-CC3): the doubles residual is the AO-direct CCSD one, and the
  # triples machinery works from 3-index integrals fitted for the SYSTEM (`cc.usedf`, the default)
  # -- no MO integral set is needed. The `usedf=false` decomposition route needs an MO dump and is
  # guarded in `calc_ccsdt`.
  has_prefix(ecm, "SVD") && ecm.exclevel[3] != :none && return true
  return ecm.exclevel[3] in (:none, :pert) && name in ("CCSD", "DCSD", "CCD", "DCD")
end


"""
    derive_mo_basis!(EC::ECInfo; persistent=false)

  Derive an MO-basis `EC.fd` from the exact AO integrals (the ± supermatrix store, `"h_AA"`,
  see `ao_integrals`) and the current orbitals, reducing the dump to the active
  space so that the MO integrals (and all downstream methods) scale with the active space
  rather than `nao` — the non-DF analogue of `dfdump`:

  - The frozen orbitals are selected exactly like for a `dfdump` ([`freeze_orbitals!`](@ref)):
    chemical/explicit frozen core, and class-honored deleted (incl. linearly-dependent)
    or explicitly frozen virtuals.
  - **Deleted and frozen virtual** orbitals are excluded from the transform — they carry no
    electrons, so no folding is needed and there is no reason to pay the O(N⁵) transform.
  - **Frozen occupied** orbitals are excluded from the transform too: their mean field is built
    directly in the AO basis (the same `2J−K` / `J(Dα+Dβ)−K(Dσ)` core Fock the AO-direct setup
    uses) and folded into `int0`/`int1` by `generate_mo_dump`.
  - `orig_orbs` records the active↔full orbital map, so user orbital lists (`occa` etc.) and
    property post-processing interpret the reduced dump correctly; the freeze options are
    **not** modified.

  The derived dump is transient by default: the driver discards it at the end of the run, and it
  is re-derived from the AO files on demand. With `persistent=true` ([`mo_integrals`](@ref), the
  `@moints` macro) the dump and its scratch file survive the run and are the user's to refresh.
"""
function derive_mo_basis!(EC::ECInfo{T}; persistent::Bool=false) where {T}
  setup_space_system!(EC; verbose=false)
  space_save = save_space(EC)
  nocc_full = length(EC.space['o'])
  nvirt_full = length(EC.space['v'])
  full_norb = length(space_save[':'])
  # the correlation reference (projected + re-orthonormalized across a geometry/basis change)
  # and the classes describing THAT set.
  # `start=dump4core_only`: the correlating orbitals then come from `start` and the frozen core is
  # swapped in from `dump` below, as in the density-fitted builder and `ao_cc_setup!`
  cMO, corr_classes = load_orbitals_for_correlation(EC; start=EC.options.wf.dump4core_only)
  # frozen core, redundant orbitals, and (dump-deleted / explicit) virtuals, all by class/index
  cls = freeze_orbitals!(EC; classes=corr_classes)
  (cls.occ_a == cls.occ_b && cls.virt_a == cls.virt_b) ||
    error("AO→MO integral transformation requires symmetric (restricted-like) freezing!")
  ncore_orbs = nocc_full - length(EC.space['o'])
  nfrozvirt = nvirt_full - length(EC.space['v'])
  frozen_occ = sort!(setdiff(space_save['o'], EC.space['o']))
  # The kept columns by their ACTUAL indices: freeze_orbitals! removes region/redundant
  # "Deleted" virtuals mid-range (never by top index), so a trailing-truncation range would
  # transform the deleted columns and drop valid high virtuals.
  active_cols = sort!(vcat(EC.space['o'], EC.space['v']))
  restore_space!(EC, space_save)
  # Transform only the kept orbitals: active + frozen-occupied (the frozen-occ are needed to
  # fold the core energy/Fock); deleted and frozen-virtual orbitals (the highest columns, see
  # `dfdump`) are dropped from the transform. Restricted orbitals build a closed-shell dump,
  # unrestricted a UHF dump (both spins keep the same number of columns — symmetric freezing).
  # dump4core_only: replace the stale (previous-geometry) frozen core reused from `start` with the
  # fresh-HF core at the current geometry, re-orthonormalizing the correlating orbitals against it
  if EC.options.wf.dump4core_only && ncore_orbs > 0
    replace_core_from_dump!(EC, cMO, ncore_orbs)
  end
  if is_restricted(cMO)
    generate_mo_dump(EC, Matrix(cMO.α)[:, active_cols]; core=Matrix(cMO.α)[:, frozen_occ], persistent)
  else
    generate_mo_dump(EC, SpinMatrix(Matrix(cMO.α)[:, active_cols], Matrix(cMO.β)[:, active_cols]);
                     core=SpinMatrix(Matrix(cMO.α)[:, frozen_occ], Matrix(cMO.β)[:, frozen_occ]), persistent)
  end
  if ncore_orbs + nfrozvirt > 0
    # record the full-space indices of the active orbitals (as `dfdump` does), so user
    # orbital lists and property post-processing translate to this reduced dump
    EC.fd.orig_orbs = active_cols
  end
  return EC.fd
end

"""
    mo_integrals(EC::ECInfo)

  Generate MO integrals from the exact (non-density-fitted) AO integrals and store them in `EC.fd`.
  This is the entry point behind the `@moints` macro, and the non-DF counterpart of `dfdump`
  (`@dfints`): the AO integrals are generated first if they are not on file yet (as `@ints` would),
  and are then transformed to the MO basis of the current orbitals ([`derive_mo_basis!`](@ref), so
  the dump covers the active space — the frozen core is folded into `int0`/`int1`).

  Unlike the dump a correlated driver derives for itself, these integrals PERSIST for the rest of
  the session and are yours to manage: they are built from a particular set of orbitals and become
  stale if the orbitals change, and re-running `@moints` is what refreshes them.
"""
function mo_integrals(EC::ECInfo{T}) where {T}
  # empty EC.fd first: a stale MO dump is superseded here, and `ao_integrals` would otherwise warn
  # about discarding integrals that this call replaces anyway
  isempty(EC.fd) || (EC.fd = FDump{T,3}())
  pm_exists(EC) || ao_integrals(EC)   # the 1-e integrals are refreshed by `generate_mo_dump`
  derive_mo_basis!(EC; persistent=true)
  draw_endline()
  return
end

"""
    ccdriver(EC::ECInfo, method; fcidump="", occa="-", occb="-")

  Run electronic structure calculation for `EC::ECInfo` using `method::String`.

  The integrals are read from `fcidump::String`.
  If `fcidump::String` is empty, the integrals from `EC.fd` are used.
  The occupied α orbitals are given by `occa::String` (default: "-").
  The occupied β orbitals are given by `occb::String` (default: "-").
  If `occb::String` is empty, the occupied β orbitals are the same as the occupied α orbitals (closed-shell case).
  The occupation strings can be given as a `+` separated list, e.g. `occa = 1+2+3` or equivalently `1-3`. 
  Additionally, the spatial symmetry of the orbitals can be specified with the syntax `orb.sym`, e.g. `occa = "-5.1+-2.2+-4.3"`.
"""
function ccdriver(EC::ECInfo, method; fcidump="", occa="-", occb="-")
  t1 = time_ns()
  save_occs = check_occs(EC, occa, occb)
  local no_user_fd = false   # set below; the finally block must see it even on an early error
  try
  check_fcidump(EC, fcidump)
  if EC.fd.df3idx
    contract_df_integrals!(EC)
  end
  # Integral-source selection: a non-empty `EC.fd` (external FCIDUMP / generated MO dump) wins;
  # otherwise the exact AO integrals (the ± supermatrix store) are used. 
  # MP2, doubles methods (CC), and perturbative triples all run AO-direct by default
  # (the frozen core is folded into an effective 1-e Hamiltonian inside `ao_cc_setup!`);
  # set `int.ao_direct=false` to route them through a derived MO dump instead.
  # Iterative triples and FCI get a transient MO dump derived from the AO integrals (`derive_mo_basis!`).
  ecmethod = ECMethod(method)
  if has_prefix(ecmethod, "EOM") && ecmethod.exclevel[1] != :full
    error("EOM requires singles — `$(method)` has none. Use an EOM method built on a " *
          "singles-carrying reference (e.g. EOM-CCSD).")
  end
  if has_prefix(ecmethod, "QV") && need_lagrange_multipliers(EC, ecmethod)
    error("The quasi-variatonal methods do not provide Λ equations, so `$(method)` cannot be " *
          "combined with a Λ prefix, `cc.properties` or `wf.natorb` (yet).")
  end
  # Creator responsibility: orbital-dependent integrals persist only when the user created them
  # (@dfints, an fcidump file). If no integrals of any kind are available, this driver creates them
  # itself — DF gives a PER-RUN fd built from the current correlation orbitals and deleted at the end
  # of this function (so it can never go stale when orbitals change between runs); int.df=false
  # generates the ± AO store, which is orbital-independent and therefore persists.
  no_user_fd = isempty(EC.fd)
  if no_user_fd && !pm_exists(EC) && !isempty(EC.system)
    if EC.options.int.df
      dfdump(EC)
    else
      ao_integrals(EC)
    end
  end
  ao_source = isempty(EC.fd)         # false when the bootstrap above produced a DF fd
  EC.ao_direct = false
  restricted_orbs = false            # AO route only: asked once, used by both decisions below
  ao_orbitals = nothing              # ... and the loaded reference is handed to `ao_cc_setup!`
  if ao_source
    pm_exists(EC) ||
      error("No integrals found: no fcidump, no AO integrals on file, and no molecular system to " *
            "generate them from. Run @dfhf/@hf (or @dfints / provide an fcidump) first.")
    setup_space_system!(EC)
    ao_orbitals = load_orbitals_for_correlation(EC; start=EC.options.wf.dump4core_only)
    restricted_orbs = is_restricted(ao_orbitals[1])
    # "closed shell" for the RESIDUAL: equal occupations AND restricted orbitals. Unrestricted
    # orbitals make the residual unrestricted even for a closed-shell method name — the method is
    # then promoted (`checkset_unrestricted_closedshell!`), not rerouted to the MO dump.
    closed_shell = (EC.space['o'] == EC.space['O']) && restricted_orbs
    # Nothing below reroutes any more: the unsupported combinations errored above, a closed-shell
    # method on unrestricted orbitals/occupations is promoted to its unrestricted form, and the
    # AO-direct path covers deleted orbitals. Only the method itself (FCI, iterative triples,
    # Brueckner) and the user's `int.ao_direct` decide. `ao_direct_method` is asked BEFORE the
    # method is promoted below, so the route follows what the user requested.
    EC.ao_direct = ao_direct_method(ecmethod) && EC.options.int.ao_direct
    if !EC.ao_direct
      derive_mo_basis!(EC)
    end
  end
  if !EC.ao_direct
    setup_space_fd!(EC)
    closed_shell = is_closed_shell(EC)
  end

  # one line saying which integrals this run uses (the routing above is silent about it)
  if EC.ao_direct
    println("Integrals: exact AO (± store), AO-direct.")
  elseif ao_source
    println("Integrals: exact AO (± store) via a transient MO dump.")
  elseif no_user_fd
    println("Integrals: density-fitted MO, generated for this run.")
  else
    println("Integrals: MO fcidump", isempty(EC.fd.origin) ? " (user-generated)." :
            " from \"$(EC.fd.origin)\".")
  end

  # Whether the ORBITALS are UHF (only the `R` methods care — they require a non-UHF reference; an
  # unrestricted residual runs on either kind). AO-direct leaves `EC.fd` empty, so it asks the orbitals.
  unrestricted_orbs = EC.ao_direct ? !restricted_orbs : EC.fd.uhf
  # Promote a closed-shell method name to its unrestricted form where the reference demands it
  closed_shell_method = checkset_unrestricted_closedshell!(ecmethod, closed_shell, unrestricted_orbs)

  energies = OutDict()
  if EC.ao_direct
    # freezes core (fold into eff. 1-e H), builds bare f_mm/e_m/d_oovv for the residual's spin case
    EHF = ao_cc_setup!(EC; closed_shell=closed_shell_method, orbitals=ao_orbitals)
    output_E_method(EHF, "HF", "energy:"); println(); flush_output()
    energies = merge(energies, "HF"=>(EHF, "HF energy"))
  else
    energies = eval_hf_energy(EC, energies, closed_shell)
  end
  # t1 = print_time(EC, t1, "HF energy", 1)
  # calculate MP2 (also the CC start guess). `calc_MP2`/`calc_UMP2` read the bare MO integrals
  # from the AO-direct dressed-integral files (`d_oovv`/`d_OOVV`/`d_oOvV`).
  if EC.options.cc.nomp2 == 0
    energies = eval_mp2_energy(EC, energies, closed_shell_method, has_prefix(ecmethod, "R"))
    # t1 = print_time(EC, t1, "MP2", 1)
  end

  request_properties = need_correlated_properties(EC)
  need_lm = need_lagrange_multipliers(EC, ecmethod)
  defer_pert_t = need_lm && defer_lambda_perturbative_triples(ecmethod)
  rdm = nothing

  if ecmethod.theory == "MP"
    save_last_amplitudes(EC, ecmethod)
    # do nothing
  elseif ecmethod.theory == "DMRG"
    energies = eval_dmrg_groundstate(EC, energies)
    t1 = print_time(EC, t1, "DMRG", 1)
  elseif ecmethod.exclevel[2] != :none
    ecmethod_cc = defer_pert_t ? ECMethod(method_name(ecmethod; main=true)) : ecmethod
    energies = eval_cc_groundstate(EC, ecmethod_cc, energies)
    t1 = print_time(EC, t1, "ground state CC", 1)
  end

  if need_lm
    calc_lm_cc(EC, ecmethod)
    t1 = print_time(EC, t1, "CC Lagrange multipliers", 1)
    if request_properties
      rdm = CoupledCluster.calc_correlated_1rdm(EC, ecmethod)
      if EC.options.cc.properties
        dipole = correlated_dipole(EC, rdm)
        if isnothing(dipole)
          println("WARNING: Dipole moment requires stored orbitals, AO basis data, and molecular geometry.")
        else
          output_dipole(method_name(ecmethod), dipole)
          energies = add_dipole_entries(energies, method_name(ecmethod), dipole; include_method_aliases=true)
        end
      end
      write_correlated_properties!(EC, rdm)
    end
  end

  if defer_pert_t
    # The dressed 3-external blocks (`d_vovv` and friends, `nocc·nvirt³` each, forced by the Λ
    # equations) are dead here: Λ and the 1-RDM above were their last readers, and the (T) kernels
    # read no dressed block beyond the oovv class. Delete them BEFORE (T) builds its bare
    # 3-external blocks, so the two generations never occupy scratch at the same time.
    for f in ("d_vovv", "d_VOVV", "d_vOvV", "d_oVvV", "d_vvvo", "d_vvoo", "d_vvvv", "d_VVVV", "d_vVvV")
      file_exists(EC, f) && delete_file!(EC, f)
    end
    energies, t1 = add_perturbative_triples(EC, ecmethod, energies; timer=t1)
  end

  if has_prefix(ecmethod, "EOM")
    exc_energies = calc_eom(EC, ecmethod)
    energies = merge(energies, exc_energies)
    t1 = print_time(EC, t1, "EOM", 1)
  end

  draw_endline()
  return energies
  finally
    # Always, error or not: a convergence failure must not leave the next driver call believing
    # an implicitly generated (orbital-dependent) dump is user-owned, running AO-direct by
    # accident, or inheriting this run's occupation strings.
    delete_temporary_files!(EC)
    EC.ao_direct = false
    if no_user_fd
      # this driver created the fd it used (per-run DF dump or AO-derived MO dump) — it deletes
      # it: both are built from the CURRENT orbitals and must not persist into the next
      # calculation (AO-direct leaves EC.fd empty anyway).
      EC.fd = FDump{ec_eltype(EC),3}()
    end
    EC.options.wf.occa, EC.options.wf.occb = save_occs
  end
end

"""
    dfccdriver(EC::ECInfo, method)

  Run electronic structure calculation for `EC::ECInfo` using `method::String`.
  
  The integrals are calculated using density fitting.
  If `EC.fd.df3idx` is set, uses pre-existing 3-index integrals (mmL/MML) from fcidump.
"""
function dfccdriver(EC::ECInfo, method)
  if EC.fd.df3idx
    setup_space_fd!(EC)
    closed_shell = is_closed_shell(EC)
  else
    setup_space_system!(EC)
    closed_shell = (EC.space['o'] == EC.space['O'])
  end
  ecmethod = ECMethod(method)
  
  energies = OutDict()
  root_name = method_name(ecmethod, root=true)
  request_properties = need_correlated_properties(EC)
  rdm = nothing
  if EC.fd.df3idx
    energies, unrestricted_orbs = eval_df3idx_mo_integrals(EC, energies, closed_shell)
  else
    onthefly = root_name == "MP2"
    energies, unrestricted_orbs = eval_df_mo_integrals(EC, energies; save3idx=!onthefly)
  end
  t1 = time_ns()
  space_save = save_space(EC)
  # frozen core, redundant orbitals, and (dump-deleted / explicit) virtuals — all selected by
  # class/index, so a region's active orbitals are never frozen by mistake. For the df3idx path
  # the redundant orbitals are already excluded from the integrals.
  freeze_orbitals!(EC; redundant=!EC.fd.df3idx)
  t1 = print_time(EC, t1, "freeze core and virt", 2)

  closed_shell_method = checkset_unrestricted_closedshell!(ecmethod, closed_shell, unrestricted_orbs)

  main_name = method_name(ecmethod)
  if has_prefix(ecmethod, "Λ")
    error("$main_name DF Lagrange multipliers are not implemented.")
  end
  if request_properties && (root_name != "MP2" || has_prefix(ecmethod, "SOS"))
    error("Correlated properties in dfccdriver are only implemented for restricted DF-MP2.")
  end
  
  if has_prefix(ecmethod, "SVD") 
    @assert ecmethod.exclevel[3] == :none "Only doubles SVD DF at this point!"
    if !closed_shell_method
      error("Only closed-shell SVD methods implemented!")
    end
    if has_prefix(ecmethod, "EOM")
      save_use_full_t2 = EC.options.cc.use_full_t2
      EC.options.cc.use_full_t2 = true
      save_project_vovo_t2 = EC.options.cc.project_vovo_t2
      EC.options.cc.project_vovo_t2 = 1
      if !save_use_full_t2 || save_project_vovo_t2 != 1
        warnerror("SVD-EOM-DCSD requires `cc.use_full_t2=true` and `cc.project_vovo_t2=1`!")
      end 
    end
    methodname = "SVD-"*root_name
    ECC = calc_svd_dc(EC, ecmethod)
    energies = output_energy(EC, ECC, energies, methodname)
  elseif root_name == "MP2" 
    if has_prefix(ecmethod, "SOS")
      ECC = calc_df_lt_sos_mp2(EC)
      energies = output_energy(EC, ECC, energies, main_name)
    else
      ECC = calc_dfmp2(EC)
      energies = output_energy(EC, ECC, energies, main_name)
    end
  elseif root_name == "CCS"
  else
    error("$main_name DF method not implemented!")
  end

  if request_properties
    rdm = dfmp2_property_rdm(EC, ecmethod)
    if EC.options.cc.properties
      dipole = correlated_dipole(EC, rdm)
      if isnothing(dipole)
        println("WARNING: Dipole moment requires stored orbitals, AO basis data, and molecular geometry.")
      else
        output_dipole(method_name(ecmethod), dipole)
        energies = add_dipole_entries(energies, method_name(ecmethod), dipole; include_method_aliases=true)
      end
    end
    write_correlated_properties!(EC, rdm)
    t1 = print_time(EC, t1, "DF-MP2 correlated properties", 1)
  end

  if has_prefix(ecmethod, "EOM")
    if has_prefix(ecmethod, "SVD")
      calc_svd_eom(EC, ecmethod)
      t1 = print_time(EC, t1, "SVD", 1)
      EC.options.cc.use_full_t2 = save_use_full_t2
      EC.options.cc.project_vovo_t2 = save_project_vovo_t2
    else
      calc_df_eom(EC, ecmethod)
      t1 = print_time(EC, t1, "DF-EOM", 1)      
    end
  end

  delete_temporary_files!(EC)
  restore_space!(EC, space_save)
  draw_endline()
  return energies
end

function fcidriver(EC::ECInfo; occa="-", occb="-", ciphi=false)
  t1 = time_ns()
  save_occs = check_occs(EC, occa, occb)
  if EC.fd.df3idx
    contract_df_integrals!(EC)
  end
  # Creator responsibility (see `ccdriver`): with no user-provided integrals, create them here —
  # a per-run DF fd (deleted at the end of this function), or the persistent ± AO store for
  # int.df=false, from which a transient MO dump is derived below.
  no_user_fd = isempty(EC.fd)
  if no_user_fd && !pm_exists(EC) && !isempty(EC.system)
    if EC.options.int.df
      dfdump(EC)
    else
      ao_integrals(EC)
    end
  end
  # FCI always needs MO integrals: with no MO dump in `EC.fd`, derive a transient one
  # from the exact AO integral files (deleted orbitals dropped, frozen core folded).
  ao_source = isempty(EC.fd)
  if ao_source
    pm_exists(EC) ||
      error("No integrals found: no fcidump, no AO integrals on file, and no molecular system to " *
            "generate them from. Run @dfhf/@hf (or @dfints / provide an fcidump) first.")
    derive_mo_basis!(EC)
  end
  if ao_source
    println("Integrals: exact AO (± store) via a transient MO dump.")
  elseif no_user_fd
    println("Integrals: density-fitted MO, generated for this run.")
  else
    println("Integrals: MO fcidump", isempty(EC.fd.origin) ? " (user-generated)." :
            " from \"$(EC.fd.origin)\".")
  end
  setup_space_fd!(EC)
  closed_shell = is_closed_shell(EC)

  energies = OutDict()
  energies = eval_hf_energy(EC, energies, closed_shell)
  # t1 = print_time(EC, t1, "HF energy", 1)

  request_properties = need_fci_properties(EC, ciphi)
  E_FCI, fci_rdm = eval_fci(EC, energies["HF"]; ciphi=ciphi, return_rdm=request_properties)
  method = ciphi ? "CIPHI" : "FCI"
  energies = output_energy(EC, E_FCI, energies, method)
  if request_properties && !isnothing(fci_rdm)
    dipole_rdm, storage_rdm = fci_property_rdms(EC, fci_rdm[1], fci_rdm[2], closed_shell)
    if write_fci_dipole(EC, ciphi)
      dipole = correlated_dipole(EC, dipole_rdm)
      if isnothing(dipole)
        println("WARNING: Dipole moment requires stored orbitals, AO basis data, and molecular geometry.")
      else
        output_dipole(method, dipole)
        energies = add_dipole_entries(energies, method, dipole; include_method_aliases=true)
      end
    end
    write_correlated_properties!(EC, storage_rdm)
  end
  t1 = print_time(EC, t1, method, 1)

  delete_temporary_files!(EC)
  draw_endline()
  if no_user_fd
    # this driver created the fd it used (per-run DF dump or AO-derived MO dump) — it deletes it:
    # both are built from the CURRENT orbitals and must not persist into the next calculation.
    EC.fd = FDump{ec_eltype(EC),3}()
  end
  # restore occs
  EC.options.wf.occa, EC.options.wf.occb = save_occs
  return energies
end

function save_last_amplitudes(EC::ECInfo, method::ECMethod)
  if is_unrestricted(method) || has_prefix(method, "R")
    if method.exclevel[1] != :none
      T1a = read_starting_guess4amplitudes(EC, Val(1), :α)
      T1b = read_starting_guess4amplitudes(EC, Val(1), :β)
      try2save_singles!(EC, T1a, T1b)
    end
    T2a = read_starting_guess4amplitudes(EC, Val(2), :α, :α)
    T2b = read_starting_guess4amplitudes(EC, Val(2), :β, :β)
    T2ab = read_starting_guess4amplitudes(EC, Val(2), :α, :β)
    try2save_doubles!(EC, T2a, T2b, T2ab)
  else
    if method.exclevel[1] != :none
      T1 = read_starting_guess4amplitudes(EC, Val(1))
      try2save_singles!(EC, T1)
    end
    T2 = read_starting_guess4amplitudes(EC, Val(2))
    try2save_doubles!(EC, T2)
  end
end

"""
    check_occs(EC::ECInfo, occa, occb)

  Check the occupation strings `occa` and `occb` and set the corresponding options in 
  [`WfOptions`](@ref ECInfos.WfOptions).
  Return the previous values of `occa` and `occb`.
"""
function check_occs(EC::ECInfo, occa, occb)
  save_occa = EC.options.wf.occa
  save_occb = EC.options.wf.occb
  if occa != "-"
    EC.options.wf.occa = occa
  end
  if occb != "-"
    EC.options.wf.occb = occb
  end
  return save_occa, save_occb
end

"""
    check_fcidump(EC::ECInfo, fcidump)

  Read the integrals from `fcidump` if it is not empty. 
"""
function check_fcidump(EC::ECInfo, fcidump) 
  if fcidump != ""
    t1 = time_ns()
    # read fcidump intergrals
    EC.fd = read_fcidump(fcidump, ec_eltype(EC))
    t1 = print_time(EC,t1,"read fcidump",1)
  end
end

"""
    eval_hf_energy(EC::ECInfo, energies::OutDict, closed_shell)

  Evaluate the Hartree-Fock energy for the integrals in `EC.fd`.
  Return the updated `energies::OutDict` with the Hartree-Fock energy (field `HF`).
"""
function eval_hf_energy(EC::ECInfo, energies::OutDict, closed_shell; rotated=false)
  t1 = time_ns()
  if !rotated
    calc_fock_matrix(EC, closed_shell)
    EHF = calc_HF_energy(EC, closed_shell)
  else
    EHF = calc_rotated_HF_energy(EC, closed_shell)
  end
  hfname = closed_shell ? "HF" : "UHF"
  output_E_method(EHF, hfname, "energy:")
  t1 = print_time(EC, t1, "$hfname energy", 1)
  println()
  flush_output()
  if !rotated
    energies = merge(energies, "HF"=>(EHF, "$hfname energy"))
  else
    energies = merge(energies, "HF(rotated)"=>(EHF, "$hfname energy"))
  end
  return energies
end

"""
    checkset_unrestricted_closedshell!(ecmethod::ECMethod, closed_shell, unrestricted)

  Check if the method is unrestricted/closed-shell and if necessary set 
  the corresponding options in [`ECMethod`](@ref ECMethod).
  Return `closed_shell_method::Bool`.
"""
function checkset_unrestricted_closedshell!(ecmethod::ECMethod, closed_shell, unrestricted)
  if is_unrestricted(ecmethod)
    closed_shell_method = false
  elseif has_prefix(ecmethod, "R")
    closed_shell_method = false
    @assert !unrestricted "For restricted methods, the orbitals must not be UHF!"
  else
    closed_shell_method = closed_shell
    if !closed_shell_method
      set_unrestricted!(ecmethod)
    end
  end
  return closed_shell_method
end

"""
    output_energy(EC::ECInfo, En::OutDict, energies::OutDict, mname; print=true)

  Print the energy components and return the updated `energies::OutDict` with 
  correction to the correlation energy (`mname*"-correction"`, e.g., ΔMP2, if available),
  same-spin(`mname*"-SS"`), opposite-spin(`mname*"-OS"`), open-shell(`mname*"-O"`) components, 
  SCS energy (`"SCS-"*mname`), correlation energy (`mname*"c"`) and 
  the total energy (field `mname`).
"""
function output_energy(EC::ECInfo, En::OutDict, energies::OutDict, mname; print=true)
  enecor = En["E"]
  enetot = En["E"]+energies["HF"]
  if haskey(energies, "HF(rotated)")
    enetot = En["E"]+energies["HF(rotated)"]
  end
  energies_out = copy(energies)
  if print
    output_E_method(enecor, mname, "correlation energy:")
    output_E_method(enetot, mname, "total energy:      ")
    println()
  end
  if haskey(En, "E-correction")
    ecorrect = En["E"] + En["E-correction"]
    ecorrectot = En["E"] + En["E-correction"] + energies["HF"]
    if print
      output_E_method(ecorrect, mname, "corrected correlation energy:")
      output_E_method(ecorrectot, mname, "corrected total energy:    ")
      println()
    end
    push!(energies_out, mname*"-correction" => (En["E-correction"], "correction to the correlation energy")) 
    if haskey(En, "E-correction δ")
      push!(energies_out, mname*"-correction δ" => (En["E-correction δ"], "uncertainty in the correction to the correlation energy")) 
    end
  end
  if haskey(En, "Expect")
    enecor = En["Expect"]
    enetot = En["Expect"]+energies["HF"]
    if print
      output_E_method(enecor, mname, "correlation expectation energy:")
      output_E_method(enetot, mname, "total expectation energy:      ")
      println()
    end
    push!(energies_out, mname*"-expect" => (En["Expect"], "correlation expectation energy")) 
  end
  if haskey(En, "ESS") && haskey(En, "EOS") && haskey(En, "EO")
    # SCS
    push!(energies_out, mname*"-SS"=>(En["ESS"], "same-spin component to the energy"), 
                        mname*"-OS"=>(En["EOS"], "opposite-spin component to the energy"),
                        mname*"-O"=>(En["EO"], "open-shell component to the energy")) 
    methodroot = method_name(ECMethod(mname), root=true)
    # calc SCS energy (if available)
    if has_spinscalingfactor(methodroot*"_ssfac")
      # get SCS factors (e.g., mp2_ssfac, ccsd_ssfac, dcsd_ssfac)
      ssfac = get_spinscalingfactor(EC, methodroot*"_ssfac")
      osfac = get_spinscalingfactor(EC, methodroot*"_osfac")
      ofac = get_spinscalingfactor(EC, methodroot*"_ofac")
      ΔE = En["E"] - En["ESS"] - En["EOS"]
      enescs = energies["HF"] + ΔE + En["ESS"]*ssfac + En["EOS"]*osfac + En["EO"]*ofac
      if print
        output_E_method(enescs, "SCS-"*mname, "total energy:")
        println()
      end
      push!(energies_out, "SCS-"*mname=>(enescs, "SCS-$mname energy"))
    end
  end

  for (type, en, desc) in En
    if startswith(type, "ω")
      push!(energies_out, mname*type => (en, "$mname excitation energy $type"),
                          type => (en, "$mname excitation energy $type"))
    end
  end
  push!(energies_out, mname*"c"=>(enecor, "$mname correlation energy"),
                      mname=>(enetot, "$mname total energy"),
                      "Ec"=>(enecor, "$mname correlation energy"),
                      "E"=>(enetot, "$mname total energy"))
  return energies_out
end

has_spinscalingfactor(name) = hasfield(ECInfos.CcOptions, Symbol(lowercase(name))) 
get_spinscalingfactor(EC::ECInfo, name) = getfield(EC.options.cc, Symbol(lowercase(name)))::Float64

"""
    eval_mp2_energy(EC::ECInfo, energies::OutDict, closed_shell, restricted)

  Evaluate the MP2 energy for the integrals in `EC.fd`. 
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::OutDict` with 
  same-spin(`MP2-SS`), opposite-spin(`MP2-OS`), open-shell(`MP2-O`) components, 
  SCS-MP2 energy (`SCS-MP2`), correlation energy (`MP2c`) and
  the MP2 energy (field `MP2`).
"""
function eval_mp2_energy(EC::ECInfo, energies::OutDict, closed_shell, restricted)
  t1 = time_ns()
  if closed_shell
    if EC.options.wf.npositron > 0
      EMp2 = calc_posMP2(EC)
    else
      EMp2 = calc_MP2(EC)
    end
    method = "MP2"
  else
    EMp2 = calc_UMP2(EC)
    method = "UMP2"
  end
  energies = output_energy(EC, EMp2, energies, method)
  t1 = print_time(EC,t1,"MP2",1)
  if !closed_shell && restricted
    spin_project_amplitudes(EC)
    EMp2 = calc_UMP2_energy(EC)
    energies = output_energy(EC, EMp2, energies, "RMP2")
  end
  energies = output_energy(EC, EMp2, energies, "MP2", print=false)
  return energies
end

"""
    output_2d_energy(EC::ECInfo, En::OutDict, energies::OutDict, method; print=true)

  Print the energy components for 2D methods and return the updated `energies::OutDict` with 
  singlet(`"SING"*method`), triplet(`"TRIP"*method`), singlet correlation(`"SING"*method*"c"`) and 
  triplet correlation(`"TRIP"*method*"c"`) components.
"""
function output_2d_energy(EC::ECInfo, En::OutDict, energies::OutDict, method; print=true)
  enecors = En["E"] + En["EW"]
  enecort = En["E"] - En["EW"]
  enetots = enecors + energies["HF"]
  enetott = enecort + energies["HF"]
  output_E_method(enetots, method, "singlet total energy:  ")
  output_E_method(enetott, method, "triplet total energy:  ")
  output_E_method(enecors, method, "singlet correlation energy:")
  output_E_method(enecort, method, "triplet correlation energy:")
  return merge(energies, "SING"*method*"c"=>(enecors,"$method singlet correlation energy"), 
                         "TRIP"*method*"c"=>(enecort,"$method triplet correlation energy"), 
                         "SING"*method=>(enetots,"$method singlet total energy"),
                         "TRIP"*method=>(enetott,"$method triplet total energy"),
                         "Ec"=>(enecors,"$method singlet correlation energy"),
                          "E"=>(enetots,"$method singlet total energy"))
end

"""
    eval_cc_groundstate(EC::ECInfo, ecmethod::ECMethod, energies_in::OutDict; save_pert_t3=false)

  Evaluate the coupled-cluster ground-state energy for the integrals in `EC.fd`.
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::OutDict` with the correlation energy (`method*"c"`) and 
  the total energy (key `method`).
"""
function eval_cc_groundstate(EC::ECInfo, ecmethod::ECMethod, energies_in::OutDict;
                            save_pert_t3=false)
  if ecmethod.exclevel[4] != :none
    error("no quadruples implemented yet...")
  end
  energies = copy(energies_in)
  if has_prefix(ecmethod, "SVD") 
    @assert ecmethod.exclevel[3] != :none "Only triples SVD at this point!"
    return eval_svd_dc_ccsdt(EC, ecmethod, energies)
  end
  t1 = time_ns()
  main_name = method_name(ecmethod)
  ECC = calc_cc(EC, ECMethod(main_name))
  if (has_prefix(ecmethod, "O") || has_prefix(ecmethod, "B")) && has_prefix(ecmethod, "QV")
    closed_shell = is_closed_shell(EC)
    energies = eval_hf_energy(EC, energies, closed_shell; rotated=true)
  end
  if has_prefix(ecmethod, "2D")
    energies = output_2d_energy(EC, ECC, energies, main_name)
  else
    energies = output_energy(EC, ECC, energies, main_name)
  end
  t1 = print_time(EC, t1, "CC", 1)

  if has_perturbative_triples(ecmethod) && !defer_lambda_perturbative_triples(ecmethod)
    energies, t1 = add_perturbative_triples(EC, ecmethod, energies;
                                            save_pert_t3=save_pert_t3, timer=t1)
  end
  return energies
end

"""
    eval_svd_dc_ccsdt(EC::ECInfo, ecmethod::ECMethod, energies::OutDict)

  Evaluate the coupled-cluster ground-state energy for the integrals in `EC.fd` using SVD-Triples.
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::OutDict` with the correlation energy (`method*"c"`) and 
  the total energy (key `method`).
"""
function eval_svd_dc_ccsdt(EC::ECInfo, ecmethod::ECMethod, energies::OutDict)
  ecmethod0 = ECMethod("CCSD(T)")
  if EC.options.cc.skip_pert_t
    @assert !EC.options.cc.calc_t3_for_decomposition "`cc.calc_t3_for_decomposition` must be false when skipping perturbative (T) calculation!"
    ecmethod0 = ECMethod("CCSD")
  end
  if is_unrestricted(ecmethod) || has_prefix(ecmethod, "R")
    error("SVD-Triples only implemented for closed-shell methods!")
  end
  energies = eval_cc_groundstate(EC, ecmethod0, energies, save_pert_t3=EC.options.cc.calc_t3_for_decomposition)

  main_name = method_name(ecmethod)
  EHF = energies["HF"]

  t1 = time_ns()
  cc3 = (ecmethod.exclevel[3] == :pertiter)
  ECC = CoupledCluster.calc_ccsdt(EC, EC.options.cc.calc_t3_for_decomposition, cc3)
  output_E_method(ECC["E"], main_name, "correlation energy:")
  output_E_method(ECC["E"]+EHF, main_name, "total energy:      ")
  if haskey(ECC, "SVD-CCSD(T)")
    output_E_method(ECC["E"] - ECC["SVD-CCSD(T)"], "SVD-DC-CCSDT - SVD-CCSD(T):")
    energies = merge(energies, "SVD-CCSD(T)c"=>(ECC["SVD-CCSD(T)"], "SVD-CCSD(T) correlation energy"),
                    "SVD-CCSD(T)"=>(ECC["SVD-CCSD(T)"]+EHF, "SVD-CCSD(T) total energy"))
    if haskey(energies, "CCSD(T)c")
      output_E_method(ECC["SVD-CCSD(T)"] - energies["CCSD(T)c"], "SVD-CCSD(T) - CCSD(T):")
      ecorr = ECC["E"] - ECC["SVD-CCSD(T)"] + energies["CCSD(T)c"]
      output_E_method(ecorr, "(T)-corrected SVD-DC-CCSDT", "correlation energy:")
      output_E_method(ecorr + EHF, "(T)-corrected SVD-DC-CCSDT", "total energy:      ")
      energies = merge(energies, 
                    main_name*"+c"=>(ecorr, "$main_name correlation energy with SVD-CCSD(T) correction"),
                    main_name*"+"=>(ecorr+EHF, "$main_name total energy with SVD-CCSD(T) correction"))
    end
  end
  t1 = print_time(EC, t1,"SVD-T",1)
  println()
  return merge(energies, main_name*"c"=>(ECC["E"], "$main_name correlation energy"), 
                         main_name=>(ECC["E"]+EHF, "$main_name total energy"))
end

"""
    eval_df_mo_integrals(EC::ECInfo, energies::OutDict; save3idx=true)

  Evaluate the density-fitted integrals in MO basis 
  and store in the correct file.
  If `save3idx` is true, save the 3-index integrals, otherwise only the 2-index integrals.

  Return the reference energy as `HF` key in OutDict and 
  `true` if the integrals are calculated using unrestricted orbitals.
"""
function eval_df_mo_integrals(EC::ECInfo, energies::OutDict; save3idx=true)
  t1 = time_ns()
  # the correlation reference: projected AND re-orthonormalized across a geometry/basis change.
  # `load_orbitals` projects only, and unlike `generate_mo_dump` this route has no orthonormality
  # assertion — with reused orbitals it silently ran on a non-orthonormal reference
  # (measured max|CᵀSC−I| ≈ 2e-5 for a 1e-3 bohr displacement).
  cMO = load_orbitals_for_correlation(EC)[1]
  unrestricted = !is_restricted(cMO)
  ERef = generate_DF_integrals(EC, cMO; save3idx)
  t1 = print_time(EC, t1, "generate DF integrals", 2)
  cMO = nothing
  output_E_method(ERef, "Reference energy:")
  println()
  return merge(energies, "HF"=>(ERef,"Reference energy")), unrestricted
end

"""
    eval_df3idx_mo_integrals(EC::ECInfo, energies::OutDict, closed_shell)

  Build the Fock matrix and reference energy from pre-existing 3-index
  MO integrals (mmL/MML) and fcidump one-electron integrals.

  Return the reference energy as `HF` key in OutDict and 
  `true` if the integrals use unrestricted orbitals.
"""
function eval_df3idx_mo_integrals(EC::ECInfo, energies::OutDict, closed_shell)
  t1 = time_ns()
  mmLfile, mmL = mmap3idx(EC, "mmL")
  SP = EC.space
  if closed_shell
    fock = gen_df3idx_fock(EC, EC.fd.int1, mmL, collect(SP['o']))
    save!(EC, "f_mm", fock)
    save!(EC, "f_MM", fock)
    eps = diag(fock)
    println("Occupied orbital energies: ", real.(eps[SP['o']]))
    save!(EC, "e_m", eps)
    save!(EC, "e_M", eps)
    ERef = real(sum(eps[SP['o']]) + sum(diag(EC.fd.int1)[SP['o']]) + EC.fd.int0)
  else
    h1a = EC.fd.uhf ? EC.fd.int1a : EC.fd.int1
    h1b = EC.fd.uhf ? EC.fd.int1b : EC.fd.int1
    has_MML = file_exists(EC, "MML")
    if has_MML
      MMLfile, MML = mmap3idx(EC, "MML")
    else
      MML = mmL
    end
    fock = gen_df3idx_fock(EC, h1a, h1b, mmL, MML, collect(SP['o']), collect(SP['O']))
    save!(EC, "f_mm", fock.α)
    save!(EC, "f_MM", fock.β)
    epsa = diag(fock.α)
    epsb = diag(fock.β)
    println("Occupied α orbital energies: ", real.(epsa[SP['o']]))
    println("Occupied β orbital energies: ", real.(epsb[SP['O']]))
    save!(EC, "e_m", epsa)
    save!(EC, "e_M", epsb)
    ERef = real(0.5 * (sum(epsa[SP['o']]) + sum(diag(h1a)[SP['o']]) 
                 + sum(epsb[SP['O']]) + sum(diag(h1b)[SP['O']])) + EC.fd.int0)
    if has_MML
      close(MMLfile)
    end
  end
  close(mmLfile)
  output_E_method(ERef, "Reference energy:")
  println()
  t1 = print_time(EC, t1, "Fock matrix and reference energy", 2)
  return merge(energies, "HF"=>(ERef,"Reference energy")), EC.fd.uhf
end

function eval_fci(EC::ECInfo, ref_energy; ciphi=false, return_rdm=false)
  t1 = time_ns()
  # Create basic FCI setup
  norb = length(EC.space[':'])
  nalpha = length(EC.space['o'])
  nbeta = length(EC.space['O'])
  ms2 = nalpha - nbeta
  nelec = nalpha + nbeta
  simtra = is_similarity_transformed(EC.fd)
  fdump = FDump{ec_eltype(EC),4}(norb, nelec, ms2=ms2, uhf=EC.fd.uhf, simtra=simtra)
  fdump.int0 = EC.fd.int0
  if EC.fd.uhf
    fdump.int1a = EC.fd.int1a
    fdump.int1b = EC.fd.int1b
    fdump.int2aa = ints2(EC, "mmmm")
    fdump.int2bb = ints2(EC, "MMMM")
    fdump.int2ab = ints2(EC, "mMmM")
  else
    fdump.int1 = EC.fd.int1
    fdump.int2 = ints2(EC, "mmmm")

    # fdump.int1 = permutedims(EC.fd.int1, (2,1))
    # fdump.int2 = permutedims(ints2(EC, "mmmm"), (3,4,1,2))
  end
  
  # Branch: Use lightweight CIPHIContext for CIPHI, full FCIContext for FCI
  if ciphi
    println("Setting up CIPHI (lightweight context)..."); flush(stdout)
    
    # Check for starting determinants from previous calculation
    nstates = EC.options.ciphi.nstates
    
    if norb < 64
      ciphi_ctx = CIPHIContext{UInt64}(fdump, EC.options.ciphi; occa=EC.space['o'], occb=EC.space['O'])
      start_dets, start_coeffs, has_start = try_fetch_starting_determinants(EC; OPattern=UInt64, nstates=nstates)
      initial_dets = has_start ? start_dets : nothing
      initial_coeffs = has_start ? start_coeffs : nothing
      E_CIPHI, coefs, dets, pt2 = run_ciphi!(ciphi_ctx; initial_dets=initial_dets, initial_coeffs=initial_coeffs)
    elseif norb < 128
      ciphi_ctx = CIPHIContext{UInt128}(fdump, EC.options.ciphi; occa=EC.space['o'], occb=EC.space['O'])
      start_dets, start_coeffs, has_start = try_fetch_starting_determinants(EC; OPattern=UInt128, nstates=nstates)
      initial_dets = has_start ? start_dets : nothing
      initial_coeffs = has_start ? start_coeffs : nothing
      E_CIPHI, coefs, dets, pt2 = run_ciphi!(ciphi_ctx; initial_dets=initial_dets, initial_coeffs=initial_coeffs)
    else
      error("CIPHIContext only implemented for norb < 128 at this point!")
    end
    t1 = print_time(EC, t1, "CIPHI", 1)
    Egs = E_CIPHI[1]
    energies = OutDict()
    for i = 1:length(E_CIPHI)-1
      energies["ω$i"] = E_CIPHI[i+1] - Egs
      if EC.options.ciphi.compute_pt2
        energies["ω$i+pt2"] = E_CIPHI[i+1] + pt2[i+1][1] - Egs - pt2[1][1]
        energies["ω$(i)δpt2"] = pt2[i+1][2]
      end
    end
    if EC.options.ciphi.compute_pt2
      push!(energies, "E-correction" => pt2[1][1])
      push!(energies, "E-correction δ" => pt2[1][2])
    end
    rdm = nothing
    if return_rdm && size(coefs, 2) > 0
      rdm_a = zeros(eltype(coefs), norb, norb)
      rdm_b = zeros(eltype(coefs), norb, norb)
      FCI.make_selected_1rdms!(rdm_a, rdm_b, dets, @view(coefs[:, 1]), norb)
      rdm = (rdm_a, rdm_b)
    end
    # Store determinants if wf.store is set
    nstates = length(E_CIPHI)
    dump_wavefunction_with_determinants!(EC, dets, coefs; nstates=nstates)
    return merge(energies, "E" => Egs - ref_energy), rdm
  else
    println("Setting up FCI..."); flush(stdout)
    compute_rdms = EC.options.fci.compute_rdms
    EC.options.fci.compute_rdms = compute_rdms || return_rdm
    fci_ctx = FCIContext(fdump, EC.options.fci; occa=EC.space['o'], occb=EC.space['O'])
    println("FCI context setup complete."); flush(stdout)
    E_FCI = run_fci!(fci_ctx)
    t1 = print_time(EC, t1, "FCI", 1)
    Egs = E_FCI[1]
    energies = OutDict()
    for i = 1:length(E_FCI)-1
      energies["ω$i"] = E_FCI[i+1] - Egs
    end
    rdm = return_rdm ? (copy(fci_ctx.rdm1_a), copy(fci_ctx.rdm1_b)) : nothing
    EC.options.fci.compute_rdms = compute_rdms
    return merge(energies, "E" => Egs - ref_energy), rdm
  end
end

"""
    eval_dmrg_groundstate(EC::ECInfo, energies::OutDict)

  Evaluate the DMRG ground-state energy for the integrals in `EC.fd`.
  HF energy must be calculated before.
  Return the updated `energies::OutDict` with the correlation energy (`"DMRGc"`) and 
  the total energy (key `"DMRG"`).
"""
function eval_dmrg_groundstate(EC::ECInfo, energies::OutDict)
  t1 = time_ns()
  ECC = calc_dmrg_dispatch(EC)
  energies = output_energy(EC, ECC, energies, "DMRG")
  t1 = print_time(EC, t1,"DMRG",1)
  return energies
end

"""
    extrapolate(energies1::OutDict, energies2::OutDict)

  Extrapolate energies using two sets of energies with corresponding corrections.

  The keys with suffix `"-correction"` are used for extrapolation.
  Return a new `OutDict` with the extrapolated energies.
  Extrapolation is done to the limit where the correction goes to zero.
"""
function extrapolate(energies1::OutDict, energies2::OutDict)
  extrapolated_energies = OutDict()
  for (key, val, desc) in energies1
    if endswith(key, "-correction")
      name = replace(key, "-correction"=>"")
      if haskey(energies2, key) && haskey(energies1, name) && haskey(energies2, name)
        e1 = energies1[name]
        e2 = energies2[name]
        c1 = val
        c2 = energies2[key]
        eex = (e2 * c1 - e1 * c2) / (c1 - c2)
        d1 = energies1(name)
        push!(extrapolated_energies, name => (eex, d1*" (extrapolated)"))
      end
    end
  end
  return extrapolated_energies
end


end #module
