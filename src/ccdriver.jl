module CCDriver
using Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.CCTools
using ..ElemCo.CoupledCluster
using ..ElemCo.FciDump

export ccdriver

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
  save_occs = check_occs(EC, occa, occb)
  check_fcidump(EC, fcidump)
  setup_space_fd!(EC)
  closed_shell = is_closed_shell(EC)

  energies = NamedTuple()
  energies = eval_hf_energy(EC, energies, closed_shell)

  ecmethod = ECMethod(method)
  closed_shell_method = checkset_unrestricted_closedshell!(EC, ecmethod, closed_shell)
  # calculate MP2
  if EC.options.cc.nomp2 == 0
    energies = eval_mp2_energy(EC, energies, closed_shell_method, has_prefix(ecmethod, "R"))
  end

  if ecmethod.theory != "MP"
    energies = eval_cc_groundstate(EC, ecmethod, energies)
  end

  if EC.options.cc.properties
    calc_lm_cc(EC, ecmethod)
  end
  delete_temporary_files!(EC)
  draw_endline()
  # restore occs
  EC.options.wf.occa, EC.options.wf.occb = save_occs
  return energies
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
    eval_hf_energy(EC::ECInfo, energies::NamedTuple, closed_shell)

  Evaluate the Hartree-Fock energy for the integrals in `EC.fd`.
  Return the updated `energies::NamedTuple` with the Hartree-Fock energy (field `HF`).
"""
function eval_hf_energy(EC::ECInfo, energies::NamedTuple, closed_shell)
  t1 = time_ns()
  calc_fock_matrix(EC, closed_shell)
  EHF = calc_HF_energy(EC, closed_shell)
  hfname = closed_shell ? "HF" : "UHF"
  @printf "%s energy: %16.12f \n" hfname EHF
  t1 = print_time(EC, t1, "$hfname energy", 1)
  println()
  flush(stdout)
  return (; energies..., HF=EHF)
end

"""
    checkset_unrestricted_closedshell!(EC::ECInfo, ecmethod::ECMethod, closed_shell)

  Check if the method is unrestricted/closed-shell and if necessary set 
  the corresponding options in [`ECMethod`](@ref ECMethod).
  Return `closed_shell_method::Bool`.
"""
function checkset_unrestricted_closedshell!(EC::ECInfo, ecmethod::ECMethod, closed_shell)
  if is_unrestricted(ecmethod)
    closed_shell_method = false
  elseif has_prefix(ecmethod, "R")
    closed_shell_method = false
    @assert !EC.fd.uhf "For restricted methods, the FCIDUMP must not be UHF!"
  else
    closed_shell_method = closed_shell
    if !closed_shell_method
      set_unrestricted!(ecmethod)
    end
  end
  return closed_shell_method
end

"""
    output_energy(EC::ECInfo, En::NamedTuple, energies::NamedTuple, mname; print=true)

  Print the energy components and return the updated `energies::NamedTuple` with 
  same-spin(`mname*"SS"`), opposite-spin(`mname*"OS"`), open-shell(`mname*"O"`) components, 
  SCS energy (`"SCS"*mname`), correlation energy (`mname*"c"`) and 
  the total energy (field `mname`) (with `-` in `mname` replaced by `_`).
"""
function output_energy(EC::ECInfo, En::NamedTuple, energies::NamedTuple, mname; print=true)
  meth = replace(mname, "-" => "_")
  enecor = En.E
  enetot = En.E+energies.HF
  if print
    @printf "%s correlation energy: \t%16.12f \n" mname enecor
    @printf "%s total energy:       \t%16.12f \n" mname enetot
    println()
    flush(stdout)
  end
  energies = (; energies..., Symbol(meth*"SS")=>En.ESS, Symbol(meth*"OS")=>En.EOS, Symbol(meth*"O")=>En.EO) 
  methodroot = replace(method_name(ECMethod(mname), root=true), "-" => "_")
  # calc SCS energy (if available)
  if hasfield(ECInfos.CcOptions, Symbol(lowercase(methodroot)*"_ssfac"))
    # get SCS factors (e.g., mp2_ssfac, ccsd_ssfac, dcsd_ssfac)
    ssfac = getfield(EC.options.cc, Symbol(lowercase(methodroot)*"_ssfac"))
    osfac = getfield(EC.options.cc, Symbol(lowercase(methodroot)*"_osfac"))
    ofac = getfield(EC.options.cc, Symbol(lowercase(methodroot)*"_ofac"))
    enescs = energies.HF + En.ESS*ssfac + En.EOS*osfac + En.EO*ofac
    if print
      @printf "SCS-%s total energy: \t%16.12f \n" mname enescs
      println()
      flush(stdout)
    end
    energies = (; energies..., Symbol("SCS"*meth)=>enescs)
  end
  return (; energies..., Symbol(meth*"c")=>enecor, Symbol(meth)=>enetot)
end

"""
    eval_mp2_energy(EC::ECInfo, energies::NamedTuple, closed_shell, restricted)

  Evaluate the MP2 energy for the integrals in `EC.fd`. 
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::NamedTuple` with 
  same-spin(`MP2SS`), opposite-spin(`MP2OS`), open-shell(`MP2O`) components, 
  SCS-MP2 energy (`SCSMP2`), correlation energy (`MP2c`) and
  the MP2 energy (field `MP2`).
"""
function eval_mp2_energy(EC::ECInfo, energies::NamedTuple, closed_shell, restricted)
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
    output_2d_energy(EC::ECInfo, En, energies::NamedTuple, method; print=true)

  Print the energy components for 2D methods and return the updated `energies::NamedTuple` with 
  singlet(`"SING"*method`), triplet(`"TRIP"*method`), singlet correlation(`"SING"*method*"c"`) and 
  triplet correlation(`"TRIP"*method*"c"`) components (with `-` in `method` replaced by `_`).
"""
function output_2d_energy(EC::ECInfo, En, energies::NamedTuple, method; print=true)
  meth = replace(method, "-" => "_")
  enecors = En.E + En.EW
  enecort = En.E - En.EW
  enetots = enecors + energies.HF
  enetott = enecort + energies.HF
  @printf "%s singlet total energy:   \t%16.12f \n" method enetots
  @printf "%s triplet total energy:   \t%16.12f \n" method enetott
  @printf "%s singlet correlation energy: \t%16.12f \n" method enecors
  @printf "%s triplet correlation energy: \t%16.12f \n" method enecort
  return (; energies..., Symbol("SING"*meth*"c")=>enecors, Symbol("TRIP"*meth*"c")=>enecort, 
          Symbol("SING"*meth)=>enetots, Symbol("TRIP"*meth)=>enetott)
end

"""
    eval_cc_groundstate(EC::ECInfo, ecmethod::ECMethod, energies::NamedTuple; save_pert_t3=false)

  Evaluate the coupled-cluster ground-state energy for the integrals in `EC.fd`.
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::NamedTuple` with the correlation energy (`method*"c"`) and 
  the total energy (field `method`) (with `-` in `method` replaced by `_`).
"""
function eval_cc_groundstate(EC::ECInfo, ecmethod::ECMethod, energies::NamedTuple;
                            save_pert_t3=false)
  if ecmethod.exclevel[4] != :none
    error("no quadruples implemented yet...")
  end
  if has_prefix(ecmethod, "SVD") 
    @assert ecmethod.exclevel[3] != :none "Only triples SVD at this point!"
    return eval_svd_dc_ccsdt(EC, ecmethod, energies)
  end
  t1 = time_ns()
  EHF = energies.HF
  main_name = method_name(ecmethod)
  ECC = calc_cc(EC, ECMethod(main_name))
  if has_prefix(ecmethod, "2D")
    energies = output_2d_energy(EC, ECC, energies, main_name)
  else
    energies = output_energy(EC, ECC, energies, main_name)
  end
  t1 = print_time(EC, t1,"CC",1)

  if has_prefix(ecmethod, "Λ")
    calc_lm_cc(EC, ecmethod)
    t1 = print_time(EC, t1,"ΛCC",1)
  end

  if ecmethod.exclevel[3] ∈ [ :pert, :pertiter]
    ET3, ET3b = calc_pertT(EC, ecmethod; save_t3=save_pert_t3)
    println()
    @printf "%s[T] total energy: %16.12f \n" main_name ECC.E+ET3b+EHF
    @printf "%s(T) correlation energy: %16.12f \n" main_name ECC.E+ET3
    @printf "%s(T) total energy: %16.12f \n" main_name ECC.E+ET3+EHF
    println()
    energies = (; energies..., T3b=ET3b, T3=ET3, 
          Symbol(main_name*"_Tc")=>ECC.E+ET3, Symbol(main_name*"_T")=>ECC.E+ET3+EHF)
  end
  return energies
end

"""
    eval_svd_dc_ccsdt(EC::ECInfo, ecmethod::ECMethod, energies::NamedTuple)

  Evaluate the coupled-cluster ground-state energy for the integrals in `EC.fd` using SVD-Triples.
  Fock matrix and HF energy must be calculated before.
  Return the updated `energies::NamedTuple` with the correlation energy (`method*"c"`) and 
  the total energy (field `method`) (with `-` in `method` replaced by `_`).
"""
function eval_svd_dc_ccsdt(EC::ECInfo, ecmethod::ECMethod, energies::NamedTuple)
  ecmethod0 = ECMethod("CCSD(T)")
  if is_unrestricted(ecmethod) || has_prefix(ecmethod, "R")
    error("SVD-Triples only implemented for closed-shell methods!")
  end
  energies = eval_cc_groundstate(EC, ecmethod0, energies, save_pert_t3=EC.options.cc.calc_t3_for_decomposition)

  main_name = method_name(ecmethod)
  EHF = energies.HF

  t1 = time_ns()
  cc3 = (ecmethod.exclevel[3] == :pertiter)
  ECC = CoupledCluster.calc_ccsdt(EC, EC.options.cc.calc_t3_for_decomposition, cc3)
  @printf "%s correlation energy: %16.12f \n" main_name ECC.E
  @printf "%s total energy: %16.12f \n" main_name ECC.E+EHF
  t1 = print_time(EC, t1,"SVD-T",1)
  println()
  meth = replace(main_name, "-" => "_")
  return (; energies..., Symbol(meth*"c")=>ECC.E, Symbol(meth)=>ECC.E+EHF)
end

end #module