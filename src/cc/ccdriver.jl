"""
    CCDriver

Module for coupled-cluster drivers.
"""
module CCDriver
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.TensorTools
using ..ElemCo.DFTools
using ..ElemCo.CCTools
using ..ElemCo.CoupledCluster
using ..ElemCo.DMRG
using ..ElemCo.DFCoupledCluster
using ..ElemCo.FciDumps
using ..ElemCo.OrbTools

export ccdriver, dfccdriver

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
  check_fcidump(EC, fcidump)
  setup_space_fd!(EC)
  closed_shell = is_closed_shell(EC)

  energies = OutDict()
  energies = eval_hf_energy(EC, energies, closed_shell)
  # t1 = print_time(EC, t1, "HF energy", 1)
  ecmethod = ECMethod(method)
  unrestricted_orbs = EC.fd.uhf
  closed_shell_method = checkset_unrestricted_closedshell!(ecmethod, closed_shell, unrestricted_orbs)
  # calculate MP2
  if EC.options.cc.nomp2 == 0
    energies = eval_mp2_energy(EC, energies, closed_shell_method, has_prefix(ecmethod, "R"))
    # t1 = print_time(EC, t1, "MP2", 1)
  end

  if ecmethod.theory == "MP"
    save_last_amplitudes(EC, ecmethod)
    # do nothing
  elseif ecmethod.theory == "DMRG"
    energies = eval_dmrg_groundstate(EC, energies)
    t1 = print_time(EC, t1, "DMRG", 1)
  else
    energies = eval_cc_groundstate(EC, ecmethod, energies)
    t1 = print_time(EC, t1, "ground state CC", 1)
  end

  if EC.options.cc.properties
    calc_lm_cc(EC, ecmethod)
    t1 = print_time(EC, t1, "CC Lagrange multipliers", 1)
  end
  delete_temporary_files!(EC)
  draw_endline()
  # restore occs
  EC.options.wf.occa, EC.options.wf.occb = save_occs
  return energies
end

"""
    dfccdriver(EC::ECInfo, method)

  Run electronic structure calculation for `EC::ECInfo` using `method::String`.
  
  The integrals are calculated using density fitting.
"""
function dfccdriver(EC::ECInfo, method)
  setup_space_system!(EC)
  closed_shell = (EC.space['o'] == EC.space['O'])
  ecmethod = ECMethod(method)
  
  energies = OutDict()
  root_name = method_name(ecmethod, root=true)
  onthefly = root_name == "MP2" 
  energies, unrestricted_orbs = eval_df_mo_integrals(EC, energies; save3idx=!onthefly)
  t1 = time_ns()
  space_save = save_space(EC)
  freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc)
  freeze_nvirt!(EC, EC.options.wf.freeze_nvirt)
  t1 = print_time(EC, t1, "freeze core and virt", 2)

  closed_shell_method = checkset_unrestricted_closedshell!(ecmethod, closed_shell, unrestricted_orbs)

  main_name = method_name(ecmethod)
  if has_prefix(ecmethod, "SVD")
    @assert ecmethod.exclevel[3] == :none "Only doubles SVD DF at this point!"
    if !closed_shell_method
      error("Only closed-shell SVD methods implemented!")
    end
    ECC = calc_svd_dc(EC, ecmethod)
    energies = output_energy(EC, ECC, energies, main_name)
  elseif root_name == "MP2"
    ECC = calc_dfmp2(EC)
    energies = output_energy(EC, ECC, energies, main_name)
  else
    error("$main_name DF method not implemented!")
  end

  delete_temporary_files!(EC)
  restore_space!(EC, space_save)
  draw_endline()
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
    EC.fd = read_fcidump(fcidump)
    t1 = print_time(EC,t1,"read fcidump",1)
  end
end

"""
    eval_hf_energy(EC::ECInfo, energies::OutDict, closed_shell)

  Evaluate the Hartree-Fock energy for the integrals in `EC.fd`.
  Return the updated `energies::OutDict` with the Hartree-Fock energy (field `HF`).
"""
function eval_hf_energy(EC::ECInfo, energies::OutDict, closed_shell)
  t1 = time_ns()
  calc_fock_matrix(EC, closed_shell)
  EHF = calc_HF_energy(EC, closed_shell)
  hfname = closed_shell ? "HF" : "UHF"
  output_E_method(EHF, hfname, "energy:")
  t1 = print_time(EC, t1, "$hfname energy", 1)
  println()
  flush_output()
  return merge(energies, "HF"=>(EHF, "$hfname energy"))
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
    EMp2 = calc_MP2(EC)
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
  EHF = energies["HF"]
  main_name = method_name(ecmethod)
  ECC = calc_cc(EC, ECMethod(main_name))
  if has_prefix(ecmethod, "2D")
    energies = output_2d_energy(EC, ECC, energies, main_name)
  else
    energies = output_energy(EC, ECC, energies, main_name)
  end
  t1 = print_time(EC, t1, "CC", 1)

  if has_prefix(ecmethod, "Λ")
    calc_lm_cc(EC, ecmethod)
    t1 = print_time(EC, t1, "ΛCC", 1)
  end

  if ecmethod.exclevel[3] ∈ [ :pert, :pertiter]
    if is_similarity_transformed(EC.fd) && !has_prefix(ecmethod, "Λ")
      warn("Perturbative triples for similarity transformed Hamiltonians must be calculated
      with ΛCCSD(T) method! The error can be ignored by setting the option `cc.ignore_error=true`.",
      !EC.options.cc.ignore_error)
    end
    ET3, ET3b = values(calc_pertT(EC, ecmethod; save_t3=save_pert_t3))
    println()
    output_E_method(ECC["E"]+ET3b+EHF, main_name*"[T]", "total energy:      ")
    output_E_method(ECC["E"]+ET3, main_name*"(T)", "correlation energy:")
    output_E_method(ECC["E"]+ET3+EHF, main_name*"(T)", "total energy:       ")
    println()
    t1 = print_time(EC, t1, "(T)", 1)
    push!(energies, "[T]"=>(ET3b,"[T] energy contribution"), 
                    "(T)"=>(ET3,"(T) energy contribution"),
                    main_name*"(T)c"=>(ECC["E"]+ET3,"$main_name(T) correlation energy"),
                    main_name*"(T)"=>(ECC["E"]+ET3+EHF,"$main_name(T) total energy"))
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
    output_E_method(ECC["SVD-CCSD(T)"] - energies["CCSD(T)c"], "SVD-CCSD(T) - CCSD(T):")
    ecorr = ECC["E"] - ECC["SVD-CCSD(T)"] + energies["CCSD(T)c"]
    output_E_method(ecorr, "(T)-corrected SVD-DC-CCSDT", "correlation energy:")
    output_E_method(ecorr + EHF, "(T)-corrected SVD-DC-CCSDT", "total energy:      ")
    energies = merge(energies, "SVD-CCSD(T)c"=>(ECC["SVD-CCSD(T)"], "SVD-CCSD(T) correlation energy"),
                    "SVD-CCSD(T)"=>(ECC["SVD-CCSD(T)"]+EHF, "SVD-CCSD(T) total energy"),
                    main_name*"+c"=>(ecorr, "$main_name correlation energy with SVD-CCSD(T) correction"),
                    main_name*"+"=>(ecorr+EHF, "$main_name total energy with SVD-CCSD(T) correction"))
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
  cMO = load_orbitals(EC, EC.options.wf.orb)
  unrestricted = !is_restricted(cMO)
  ERef = generate_DF_integrals(EC, cMO; save3idx)
  t1 = print_time(EC, t1, "generate DF integrals", 2)
  cMO = nothing
  output_E_method(ERef, "Reference energy:")
  println()
  return merge(energies, "HF"=>(ERef,"Reference energy")), unrestricted
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
  ext = Base.get_extension(@__MODULE__, :DmrgExt)
  if isnothing(ext)
    ECC = calc_dmrg()
  else
    ECC = ext.calc_dmrg(EC)
  end
  energies = output_energy(EC, ECC, energies, "DMRG")
  t1 = print_time(EC, t1,"DMRG",1)
  return energies
end

end #module
