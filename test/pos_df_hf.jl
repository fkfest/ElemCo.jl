using ElemCo

@testset "Positron DF-HF Test" begin

  epsilon = 1.e-6

  # Reference energies
  EHF_H_ref        = -0.660770127162853
  Ecorr_MP2_H_ref  = 0.008491457275
  EHF_LiH_ref      = -7.988745934771541
  Ecorr_MP2_LiH_ref = -0.343860916981

  # Geometries
  xyz_H = """
    bohr
    H 0.0 0.0 0.0
  """

  xyz_LiH = """
    bohr
    Li 0.0  0.0  0.0
    H  0.0  0.0  3.0196
    H2 0.0 -1.0  4.0196
    H2 0.0  1.0  4.0196
  """

  # Basis sets
  basis_H = Dict(
    "ao"    => "aug-cc-pVDZ",
    "jkfit" => "def2-universal-jkfit",
    "mpfit" => "cc-pvtz-rifit"
  )

  basis_LiH = Dict(
    "ao"    => "aug-cc-pVDZ",
    "jkfit" => "def2-universal-jkfit",
    "mpfit" => "cc-pvtz-rifit"
  )

  #---------------------------------------------------------------------------
  @testset "H⁻ + positron" begin
    EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_H, basis_H))
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
    EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_LiH, basis_LiH))
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
    EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_LiH, basis_LiH))
    @set wf charge = 0
    @set wf npositron = 1
    @set wf freeze_nocc = 0
    @set scf direct = 1
    @dummy["H2"]

    E_LiH_direct = @dfhf
    @test abs(E_LiH_direct["HF"] - EHF_LiH_ref) < epsilon
  end

end
