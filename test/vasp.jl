using ElemCo
using ElemCo.ECInfos
using ElemCo.FciDumps: headvar
using ElemCo.TensorTools: mmap3idx, closemmap
using ElemCo.IntegralTools: contract_df_integrals!
using ElemCo.VaspInterface

# cc4s reference energies (in eV) and conversion factor
const eV2Eh = 0.036749322175638754

@testset "VASP Interface Test" begin
  # Test data directories: gamma (real integrals) and shifted (complex integrals)
  gamma_dir = joinpath(@__DIR__, "files", "gamma")
  shifted_dir = joinpath(@__DIR__, "files", "shifted")

  @testset "YAML parsing (gamma)" begin
    meta = VaspInterface.read_vasp_yaml(joinpath(gamma_dir, "EigenEnergies.yaml"))
    @test meta.version == 100
    @test meta.scalar_type == :Real64
    @test meta.elements_type == :TextFile
    @test length(meta.dims) == 1
    @test meta.dims[1].length == 11
    @test meta.dims[1].type == "State"
    @test meta.unit ≈ 0.03674932217563878
    @test !meta.half_grid
    @test haskey(meta.metadata, "fermiEnergy")

    cv_meta = VaspInterface.read_vasp_yaml(joinpath(gamma_dir, "CoulombVertex.yaml"))
    @test cv_meta.scalar_type == :Complex64
    @test cv_meta.elements_type == :IeeeBinaryFile
    @test length(cv_meta.dims) == 3
    @test cv_meta.dims[1].length == 64
    @test cv_meta.dims[1].type == "AuxiliaryField"
    @test cv_meta.dims[2].length == 11
    @test cv_meta.half_grid == false
  end

  @testset "YAML parsing (shifted)" begin
    cv_meta = VaspInterface.read_vasp_yaml(joinpath(shifted_dir, "CoulombVertex.yaml"))
    @test cv_meta.scalar_type == :Complex64
    @test cv_meta.elements_type == :IeeeBinaryFile
    @test cv_meta.dims[1].length == 83
    @test cv_meta.dims[2].length == 11
    @test cv_meta.half_grid == false
  end

  @testset "load_vasp (gamma)" begin
    data = load_vasp(gamma_dir)

    @test data.n_occupied == 1
    @test data.n_virtual == 10
    @test length(data.eigen_energies) == 11

    # Coulomb vertex shape: (naux, norb, norb), no halfGrid doubling
    @test size(data.coulomb_vertex) == (64, 11, 11)

    # Optional tensors loaded
    @test length(data.coulomb_potential) == 93
    @test size(data.grid_vectors) == (3, 93)
    @test size(data.delta_integrals_hh) == (1, 1)
    @test size(data.delta_integrals_pphh) == (10, 10, 1, 1)
    @test size(data.mp2_pair_energies) == (1, 1)
  end

  @testset "load_vasp (shifted)" begin
    data = load_vasp(shifted_dir)

    @test data.n_occupied == 1
    @test data.n_virtual == 10
    @test length(data.eigen_energies) == 11
    @test size(data.coulomb_vertex) == (83, 11, 11)
  end

  @testset "setup_vasp! (gamma)" begin
    data = load_vasp(gamma_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    @test headvar(EC.fd, "NORB", Int) == 11
    @test headvar(EC.fd, "NELEC", Int) == 2
    @test headvar(EC.fd, "MS2", Int) == 0

    @test length(EC.space['o']) == 1
    @test length(EC.space['v']) == 10

    @test size(EC.fd.int1) == (11, 11)
    @test EC.fd.int1[1,1] != data.eigen_energies[1]

    # Check mmL file was created and has correct shape
    mmLfile, mmL = mmap3idx(EC, "mmL")
    @test size(mmL, 1) == 11
    @test size(mmL, 2) == 11
    @test 0 < size(mmL, 3) <= 64
    closemmap(EC, mmLfile, mmL)

    @test EC.fd.df3idx == true
    @test length(EC.fd.int2) == 0
  end

  @testset "contract_df_integrals! (gamma)" begin
    data = load_vasp(gamma_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    contract_df_integrals!(EC)

    @test EC.fd.df3idx == false
    norbs = 11
    ntri = norbs * (norbs + 1) ÷ 2
    @test size(EC.fd.int2) == (norbs, norbs, ntri)

    # Verify int2 by comparing with direct contraction from mmL
    mmLfile, mmL = mmap3idx(EC, "mmL")
    ref_1111 = sum(mmL[1,1,:] .* mmL[1,1,:])
    tri_11 = 1
    @test EC.fd.int2[1,1,tri_11] ≈ ref_1111
    ref_2312 = sum(mmL[2,1,:] .* mmL[3,2,:])
    tri_12 = 2
    @test EC.fd.int2[2,3,tri_12] ≈ ref_2312
    closemmap(EC, mmLfile, mmL)
  end

  @testset "CCSD with gamma (real integrals)" begin
    # cc4s reference energies (eV) converted to Hartree
    E_MP2_ref  = -0.27083595420491191 * eV2Eh
    E_CCSD_ref = -0.43042641733358789 * eV2Eh
    epsilon = 1.e-6

    data = load_vasp(gamma_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    energies = ElemCo.ccdriver(EC, "ccsd"; fcidump="")
    E_MP2 = real(energies["MP2c"])
    E_CCSD = real(energies["CCSDc"])
    @test isfinite(E_MP2)
    @test isfinite(E_CCSD)
    @test E_MP2 < 0.0
    @test E_CCSD < 0.0
    @test abs(E_MP2 - E_MP2_ref) < epsilon
    @test abs(E_CCSD - E_CCSD_ref) < epsilon
  end

  @testset "CCSD with shifted (complex integrals)" begin
    # cc4s reference energies (eV) converted to Hartree
    E_MP2_ref  = -0.26972030543776049 * eV2Eh
    E_CCSD_ref = -0.42763177721816037 * eV2Eh
    epsilon = 1.e-6

    data = load_vasp(shifted_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    energies = ElemCo.ccdriver(EC, "ccsd"; fcidump="")
    E_MP2 = real(energies["MP2c"])
    E_CCSD = real(energies["CCSDc"])
    @test isfinite(E_MP2)
    @test isfinite(E_CCSD)
    @test E_MP2 < 0.0
    @test E_CCSD < 0.0
    @test abs(E_MP2 - E_MP2_ref) < epsilon
    @test abs(E_CCSD - E_CCSD_ref) < epsilon
  end

  @testset "SVD-DC-CCSDT with gamma (empty triples basis)" begin
    # Regression test: complex VASP integrals + svd-dc-ccsdt on a system with
    # no significant triples (single occupied orbital) used to crash in
    # rotate_U2pseudocanonical via a 0x0 complex eigen (LAPACK ZHEEVR param 15).
    # The empty triples SVD basis is now detected and the triples are skipped,
    # so the result reduces to the CCSD energy.
    epsilon = 1.e-6
    data = load_vasp(gamma_dir)
    EC = ECInfo{eltype(data.coulomb_vertex)}()
    setup_vasp!(EC, data)

    energies = ElemCo.ccdriver(EC, "svd-dc-ccsdt"; fcidump="")
    E_CCSD = real(energies["CCSDc"])
    E_SVD = real(energies["SVD-DC-CCSDTc"])
    @test isfinite(E_SVD)
    @test abs(E_SVD - E_CCSD) < epsilon
  end
end
