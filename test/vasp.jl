using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps: headvar
using ElemCo.TensorTools: mmap3idx, closemmap
using ElemCo.VaspInterface

@testset "VASP Interface Test" begin
  vasp_dir = joinpath(@__DIR__, "files", "rBN_ref_yaml_files")

  @testset "YAML parsing" begin
    meta = VaspInterface.read_vasp_yaml(joinpath(vasp_dir, "EigenEnergies.yaml"))
    @test meta.version == 100
    @test meta.scalar_type == :Real64
    @test meta.elements_type == :TextFile
    @test length(meta.dims) == 1
    @test meta.dims[1].length == 96
    @test meta.dims[1].type == "State"
    @test meta.unit ≈ 0.03674932217563878
    @test !meta.half_grid
    @test haskey(meta.metadata, "fermiEnergy")

    cv_meta = VaspInterface.read_vasp_yaml(joinpath(vasp_dir, "CoulombVertex.yaml"))
    @test cv_meta.scalar_type == :Complex64
    @test cv_meta.elements_type == :IeeeBinaryFile
    @test length(cv_meta.dims) == 3
    @test cv_meta.dims[1].length == 356
    @test cv_meta.dims[1].type == "AuxiliaryField"
    @test cv_meta.dims[2].length == 96
    @test cv_meta.half_grid == true
  end

  @testset "load_vasp" begin
    data = load_vasp(vasp_dir)

    # Check dimensions
    @test data.n_occupied == 16
    @test data.n_virtual == 80
    @test length(data.eigen_energies) == 96

    # Eigenvalues should be in Hartree (unit ≈ 0.0367 is Eh/eV factor)
    # First eigenvalue: -25.566... eV * 0.0367... = ~ -0.9397 Eh
    @test data.eigen_energies[1] ≈ -25.5661014910229 * 0.03674932217563878

    # Coulomb vertex shape: (naux, norb, norb)
    # With halfGrid: naux = 2*356 = 712
    @test size(data.coulomb_vertex) == (712, 96, 96)

    # Optional tensors loaded
    @test length(data.coulomb_potential) == 819
    @test size(data.grid_vectors) == (3, 819)
    @test size(data.delta_integrals_hh) == (16, 16)
    @test size(data.delta_integrals_pphh) == (80, 80, 16, 16)
    @test size(data.mp2_pair_energies) == (16, 16)
  end

  @testset "setup_vasp!" begin
    data = load_vasp(vasp_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    @test headvar(EC.fd, "NORB", Int) == 96
    @test headvar(EC.fd, "NELEC", Int) == 32
    @test headvar(EC.fd, "MS2", Int) == 0

    # Check orbital spaces
    @test length(EC.space['o']) == 16
    @test length(EC.space['v']) == 80

    # Check 1-electron integrals are the core Hamiltonian h₀ (NOT the diagonal Fock)
    # h_{pp} = ε_p - 2*J_{pp} + K_{pp}, so off-diagonal elements may be nonzero
    @test size(EC.fd.int1) == (96, 96)
    # The diagonal should differ from eigenvalues (because of 2J-K subtraction)
    @test EC.fd.int1[1,1] != data.eigen_energies[1]

    # Check mmL file was created and has correct shape
    mmLfile, mmL = mmap3idx(EC, "mmL")
    @test size(mmL) == (96, 96, 712)
    closemmap(EC, mmLfile, mmL)
  end
end
