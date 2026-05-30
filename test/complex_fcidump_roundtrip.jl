using Test
using ElemCo
using ElemCo.FciDumps
using ElemCo.FciDumps: headvar
using ElemCo.QMTensors
using LinearAlgebra

# Complex integrals lack the full permutational symmetry of real integrals, so
# reading/writing them must use the same reduced symmetry as similarity-transformed
# (ST=1) fcidumps -- without setting the ST flag. This test checks that a complex
# fcidump survives a write/read round-trip even though ST=0.
@testset "Complex FCIDUMP round-trip (ST=0)" begin
  epsilon = 1.e-12

  fd_real = read_fcidump(joinpath(@__DIR__, "files", "H2O.FCIDUMP"))
  norb = headvar(fd_real, "NORB", Int)
  nelec = headvar(fd_real, "NELEC", Int)
  nocc = nelec ÷ 2

  # Diagonal phase rotation of the virtual orbitals produces genuinely complex
  # integrals that break the real 8-fold permutational symmetry.
  phases = zeros(norb)
  phases[nocc+1:end] .= 0.1
  U = diagm(exp.(im .* phases))

  fd = FDump{ComplexF64,3}(fd_real)
  transform_fcidump!(fd, SpinMatrix(conj(U)), SpinMatrix(U))

  # We deliberately do NOT mark the dump as similarity transformed.
  @test headvar(fd, "ST", Int) == 0

  tmp = tempname() * ".FCIDUMP"
  write_fcidump(fd, tmp)
  fd2 = read_fcidump(tmp, ComplexF64)
  rm(tmp; force=true)

  # The complex flag is auto-detected from the written header.
  @test headvar(fd2, "ICMPLX", Int) > 0
  @test headvar(fd2, "ST", Int) == 0

  @test maximum(abs.(fd.int2 .- fd2.int2)) < epsilon
  @test maximum(abs.(fd.int1 .- fd2.int1)) < epsilon
  @test abs(fd.int0 - fd2.int0) < epsilon
end
