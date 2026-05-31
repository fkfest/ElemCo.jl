using Test
using ElemCo
using ElemCo.TrexioInterface
using ElemCo.QMTensors
using LinearAlgebra

@testset "Complex TREXIO Round-Trip" begin

norb = 5
nocc = 2
nvirt = norb - nocc

# ============================================================
# Complex MO coefficient round-trip (restricted)
# ============================================================
@testset "Complex restricted rotations" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_rotations.h5")

  # Create complex rotation matrix
  C_real = Matrix{Float64}(I, norb, norb)
  C_real[1,2] = 0.1; C_real[2,1] = -0.1
  C_imag = zeros(norb, norb)
  C_imag[1,3] = 0.05; C_imag[3,1] = -0.05
  C = complex.(C_real, C_imag)
  rotations = SpinMatrix(C)

  # Write
  open_trexio(trexio_file, "w") do io
    write_trexio_rotations(io, rotations; type="TestComplex")
  end

  # Read
  rotations_back, type = open_trexio(trexio_file, "r") do io
    read_trexio_rotations(io; verbose=false)
  end

  @test type == "TestComplex"
  @test rotations_back[1] ≈ rotations[1]
  @test eltype(rotations_back[1]) <: Complex

  rm(tmpdir; force=true, recursive=true)
end

# ============================================================
# Complex MO coefficient round-trip (unrestricted)
# ============================================================
@testset "Complex unrestricted rotations" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_urotations.h5")

  # Create complex rotation matrices for alpha and beta
  Ca = complex.(Matrix{Float64}(I, norb, norb), 0.02 * randn(norb, norb))
  Cb = complex.(Matrix{Float64}(I, norb, norb), 0.03 * randn(norb, norb))
  rotations = SpinMatrix(Ca, Cb)

  # Write
  open_trexio(trexio_file, "w") do io
    write_trexio_rotations(io, rotations; type="UTestComplex")
  end

  # Read
  rotations_back, type = open_trexio(trexio_file, "r") do io
    read_trexio_rotations(io; verbose=false)
  end

  @test type == "UTestComplex"
  @test rotations_back[1] ≈ rotations[1]
  @test rotations_back[2] ≈ rotations[2]
  @test eltype(rotations_back[1]) <: Complex

  rm(tmpdir; force=true, recursive=true)
end

# ============================================================
# Complex restricted amplitude round-trip
# ============================================================
@testset "Complex restricted amplitudes" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_amps.h5")

  # Create complex T1 and T2 amplitudes
  T1 = complex.(0.1 * randn(nvirt, nocc), 0.05 * randn(nvirt, nocc))
  T2_full = complex.(0.01 * randn(nvirt, nvirt, nocc, nocc), 0.005 * randn(nvirt, nvirt, nocc, nocc))
  # Antisymmetrize T2: T2[a,b,i,j] = -T2[b,a,i,j] = -T2[a,b,j,i] = T2[b,a,j,i]
  T2 = T2_full - permutedims(T2_full, (2,1,3,4)) - permutedims(T2_full, (1,2,4,3)) + permutedims(T2_full, (2,1,4,3))

  # Write
  open_trexio(trexio_file, "w") do io
    write_trexio_amplitudes(io, T1, T2)
  end

  # Read
  T1_back, T2_back = open_trexio(trexio_file, "r") do io
    (read_trexio_singles(io), read_trexio_doubles(io))
  end

  @test T1_back ≈ T1
  @test eltype(T1_back) <: Complex
  @test T2_back ≈ T2
  @test eltype(T2_back) <: Complex

  rm(tmpdir; force=true, recursive=true)
end

# ============================================================
# Complex unrestricted amplitude round-trip
# ============================================================
@testset "Complex unrestricted amplitudes" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_uamps.h5")

  noccA = 3; noccB = 2
  nvirtA = norb - noccA; nvirtB = norb - noccB

  # Create complex T1a, T1b
  T1a = complex.(0.1 * randn(nvirtA, noccA), 0.05 * randn(nvirtA, noccA))
  T1b = complex.(0.1 * randn(nvirtB, noccB), 0.05 * randn(nvirtB, noccB))

  # Create complex T2a (antisymmetric), T2b (antisymmetric), T2ab (no symmetry)
  T2a_raw = complex.(0.01 * randn(nvirtA, nvirtA, noccA, noccA), 0.005 * randn(nvirtA, nvirtA, noccA, noccA))
  T2a = T2a_raw - permutedims(T2a_raw, (2,1,3,4)) - permutedims(T2a_raw, (1,2,4,3)) + permutedims(T2a_raw, (2,1,4,3))

  T2b_raw = complex.(0.01 * randn(nvirtB, nvirtB, noccB, noccB), 0.005 * randn(nvirtB, nvirtB, noccB, noccB))
  T2b = T2b_raw - permutedims(T2b_raw, (2,1,3,4)) - permutedims(T2b_raw, (1,2,4,3)) + permutedims(T2b_raw, (2,1,4,3))

  T2ab = complex.(0.01 * randn(nvirtA, nvirtB, noccA, noccB), 0.005 * randn(nvirtA, nvirtB, noccA, noccB))

  # Write
  open_trexio(trexio_file, "w") do io
    write_trexio_amplitudes(io, T1a, T1b, T2a, T2b, T2ab)
  end

  # Read
  T1a_back, T1b_back = open_trexio(trexio_file, "r") do io
    read_trexio_unrestricted_singles(io)
  end
  T2a_back, T2b_back, T2ab_back = open_trexio(trexio_file, "r") do io
    read_trexio_unrestricted_doubles(io)
  end

  @test T1a_back ≈ T1a
  @test T1b_back ≈ T1b
  @test eltype(T1a_back) <: Complex
  @test T2a_back ≈ T2a
  @test T2b_back ≈ T2b
  @test T2ab_back ≈ T2ab
  @test eltype(T2a_back) <: Complex

  rm(tmpdir; force=true, recursive=true)
end

# ============================================================
# Real amplitude round-trip still works (no _im fields)
# ============================================================
@testset "Real amplitudes backward compatibility" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_real_amps.h5")

  T1 = 0.1 * randn(nvirt, nocc)
  T2_raw = 0.01 * randn(nvirt, nvirt, nocc, nocc)
  T2 = T2_raw - permutedims(T2_raw, (2,1,3,4)) - permutedims(T2_raw, (1,2,4,3)) + permutedims(T2_raw, (2,1,4,3))

  # Write real amplitudes
  open_trexio(trexio_file, "w") do io
    write_trexio_amplitudes(io, T1, T2)
  end

  # Read back — should be real (no _im fields present)
  T1_back, T2_back = open_trexio(trexio_file, "r") do io
    (read_trexio_singles(io), read_trexio_doubles(io))
  end

  @test T1_back ≈ T1
  @test eltype(T1_back) == Float64
  @test T2_back ≈ T2
  @test eltype(T2_back) == Float64

  rm(tmpdir; force=true, recursive=true)
end

# ============================================================
# Real 1-RDM write helper
# ============================================================
@testset "Real 1-RDM write" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_rdm.h5")

  rotations = SpinMatrix(Matrix{Float64}(I, norb, norb))
  rdm = SpinMatrix(diagm(0 => [2.0, 2.0, 0.0, 0.0, 0.0]))

  open_trexio(trexio_file, "w") do io
    write_trexio_rotations(io, rotations; type="TestRDM")
    write_trexio_1rdm(io, rdm)
  end

  open_trexio(trexio_file, "r") do io
    @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
  end

  rm(tmpdir; force=true, recursive=true)
end

@testset "Unrestricted 1-RDM write" begin
  tmpdir = mktempdir()
  trexio_file = joinpath(tmpdir, "test_urdm.h5")

  rotations = SpinMatrix(Matrix{Float64}(I, norb, norb), Matrix{Float64}(I, norb, norb))
  rdm = SpinMatrix(diagm(0 => [1.0, 1.0, 1.0, 0.0, 0.0]),
                   diagm(0 => [1.0, 1.0, 0.0, 0.0, 0.0]))

  open_trexio(trexio_file, "w") do io
    write_trexio_rotations(io, rotations; type="UTestRDM")
    write_trexio_1rdm(io, rdm)
  end

  open_trexio(trexio_file, "r") do io
    @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
    @test ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
  end

  rm(tmpdir; force=true, recursive=true)
end

end
