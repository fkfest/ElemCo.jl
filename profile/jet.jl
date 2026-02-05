using ElemCo
using JET

function main()
geometry="O      0.000000000    0.000000000   -0.130186067
            H1     0.000000000    1.489124508    1.033245507
            H2     0.000000000   -1.489124508    1.033245507"

basis=Dict("ao"=>"6-31g", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvtz-mpfit")

@time @ECinit
@time @dfhf
@dfints

println("\n" * "="^80)
println("RUNNING JET TYPE STABILITY ANALYSIS")
println("="^80 * "\n")

@report_opt ignored_modules=(Base, ElemCo.Outputs, ElemCo.DMRG, ElemCo.TREXIO, ElemCo.DIIS) target_modules=(@__MODULE__,
                            ElemCo,
                            ElemCo.BOHF,
                            ElemCo.Drivers,
                            ElemCo.CCTools,
                            ElemCo.Constants,
                            ElemCo.CoupledCluster,
                            ElemCo.DecompTools,
                            ElemCo.DescDict,
                            ElemCo.DFCoupledCluster,
                            ElemCo.DfDump,
                            ElemCo.DFHF,
                            ElemCo.DFMCSCF,
                            ElemCo.DFTools,
                            #ElemCo.DIIS,
                            #ElemCo.DMRG,
                            ElemCo.DumpTools,
                            ElemCo.ECInfos,
                            ElemCo.ECMethods,
                            ElemCo.FciDumps,
                            ElemCo.FockFactory,
                            ElemCo.MIO,
                            ElemCo.MNPY,
                            ElemCo.OrbTools,
                            ElemCo.QMTensors,
                            ElemCo.TensorTools,
                            ElemCo.Utils,
                            ElemCo.Wavefunctions,
                            ElemCo.BasisSets,
                            ElemCo.Elements,
                            ElemCo.Integrals,
                            ElemCo.Interfaces,
                            ElemCo.Libcint,
                            ElemCo.MoldenInterface,
                            ElemCo.MolproInterface,
                            ElemCo.MSystems,
                            ElemCo.FCI,
                            #) ElemCo.DfDump.dfdump(EC)
                            #) ElemCo.DFHF.dfhf(EC)
                            #) ElemCo.Drivers.dfccdriver(EC,"MP2")
                            ) ElemCo.Drivers.ccdriver(EC,"λCCSD")
                            # ) ElemCo.Drivers.ccdriver(EC,"SVD-DC-CCSDT")
                            #) ElemCo.Drivers.dfccdriver(EC,"SVD-DCSD")
                            #  ) ElemCo.Drivers.fcidriver(EC,ciphi=true)
                            

end
@time main()
