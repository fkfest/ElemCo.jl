@testitem "df3idx_vasp" tags=[:interface, :quick] begin
using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps: headvar
using ElemCo.IntegralTools: contract_df_integrals!
using ElemCo.VaspInterface

@testset "DF-3IDX VASP" begin
  epsilon = 1e-6

  for (label, vasp_dir) in [("gamma", joinpath(@__DIR__, "files", "gamma")),
                             ("shifted", joinpath(@__DIR__, "files", "shifted"))]
    @testset "$label" begin
      data = load_vasp(vasp_dir)

      # Run DF-HF + DF-MP2 via the df3idx path
      EC = ECInfo{eltype(data.coulomb_vertex)}()
      setup_vasp!(EC, data)
      @assert EC.fd.df3idx

      ehf = ElemCo.dfhf(EC)
      EHF_df3idx = ehf["HF"]
      @test isfinite(real(EHF_df3idx))

      energies_df = ElemCo.dfccdriver(EC, "mp2")
      EHF_dfcc = energies_df["HF"]
      EMP2c_df = energies_df["MP2c"]
      @test abs(EHF_dfcc - EHF_df3idx) < epsilon
      @test isfinite(real(EMP2c_df))
      @test real(EMP2c_df) < 0.0

      # Cross-check: contract integrals and use ccdriver
      contract_df_integrals!(EC)
      energies_cc = ElemCo.ccdriver(EC, "mp2"; fcidump="")
      EHF_cc = energies_cc["HF"]
      EMP2c_cc = energies_cc["MP2c"]
      @test abs(EHF_cc - EHF_df3idx) < epsilon
      @test abs(EMP2c_cc - EMP2c_df) < epsilon
    end
  end
end
end
