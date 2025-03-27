using ElemCo
using JET

function main()
geometry="O      0.000000000    0.000000000   -0.130186067
            H1     0.000000000    1.489124508    1.033245507
            H2     0.000000000   -1.489124508    1.033245507"

basis="vdz"

@time @ECinit
@time @dfhf
#@time @dfcc mp2
@time @cc dcsd
#@time @dfcc svd-dcsd

@report_opt target_modules=(@__MODULE__,
                            ElemCo,
                            ElemCo.BOHF,
                            ElemCo.CCDriver,
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
                            ElemCo.DIIS,
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
                            ElemCo.Libcint5,
                            ElemCo.MoldenInterface,
                            ElemCo.MolproInterface,
                            ElemCo.MSystem
                            #) ElemCo.DfDump.dfdump(EC)
                            #) ElemCo.DFHF.dfhf(EC)
                            #) ElemCo.CCDriver.dfccdriver(EC,"MP2")
                            #) ElemCo.CCDriver.ccdriver(EC,"λCCSD")
                            ) ElemCo.CCDriver.ccdriver(EC,"SVD-DC-CCSDT")
                            #) ElemCo.CCDriver.dfccdriver(EC,"SVD-DCSD")
                            
end
@time main()
