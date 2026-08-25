@testitem "pos_df_hf" tags=[:pos, :quick] begin
using ElemCo

@testset "Positron DF-HF Test" begin

  epsilon = 1.e-6

  # Reference energies
  EHF_H_ref        = -0.660770127162853
  Ecorr_MP2_H_ref  = -0.047265317612
  EHF_LiH_ref      = -7.988745934771541
  Ecorr_MP2_LiH_ref = -0.038593138170

  #---------------------------------------------------------------------------
  @testset "H⁻ + positron" begin
    geometry = """
      bohr
      H 0.0 0.0 0.0
    """
    basis = Dict(
      "ao"    => "aug-cc-pVDZ",
      "jkfit" => "def2-universal-jkfit",
      "mpfit" => "cc-pvtz-rifit"
    )

    @ECinit
    @set wf charge = -1
    @set wf npositron = 1
    @set wf freeze_nocc = 0

    E_H = @dfhf
    @test abs(E_H["HF"] - EHF_H_ref) < epsilon

    @dfints
    E_H = @cc MP2
    @test abs(E_H["MP2c"] - Ecorr_MP2_H_ref) < epsilon
  end

  #---------------------------------------------------------------------------
  @testset "LiH + positron" begin
    geometry = """
      bohr
      Li 0.0  0.0  0.0
      H  0.0  0.0  3.0196
      H2 0.0 -1.0  4.0196
      H2 0.0  1.0  4.0196
    """
    basis = Dict(
      "ao"    => "aug-cc-pVDZ",
      "jkfit" => "def2-universal-jkfit",
      "mpfit" => "cc-pvtz-rifit"
    )

    @ECinit
    @set wf charge = 0
    @set wf npositron = 1
    @set wf freeze_nocc = 0
    @dummy["H2"]

    E_LiH = @dfhf
    @test abs(E_LiH["HF"] - EHF_LiH_ref) < epsilon

    @dfints
    E_LiH = @cc MP2
    @test abs(E_LiH["MP2c"] - Ecorr_MP2_LiH_ref) < epsilon
  end

  #---------------------------------------------------------------------------
  @testset "LiH + positron (direct SCF)" begin
    geometry = """
      bohr
      Li 0.0  0.0  0.0
      H  0.0  0.0  3.0196
      H2 0.0 -1.0  4.0196
      H2 0.0  1.0  4.0196
    """
    basis = Dict(
      "ao"    => "aug-cc-pVDZ",
      "jkfit" => "def2-universal-jkfit",
      "mpfit" => "cc-pvtz-rifit"
    )

    @ECinit
    @set wf charge = 0
    @set wf npositron = 1
    @set wf freeze_nocc = 0
    @set scf direct = 1
    @dummy["H2"]

    E_LiH_direct = @dfhf
    @test abs(E_LiH_direct["HF"] - EHF_LiH_ref) < epsilon
  end

end
end
