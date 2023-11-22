using Test

# choose what to test with `Pkg.test("ElemCo", test_args=["h2o","df_hf","all"])``
# or `$ julia runtests.jl h2o df_hf all`

runall = length(ARGS) == 0 || "all" in ARGS
if runall
    println("Running ALL tests")
else
    println("Running only $ARGS")
end

@testset verbose = true "FCIDUMP Calculations" begin

tests = ["h2o", "h2o_st1", "h2o_cation", "2d_cc"]
for test in tests
  if runall || test in ARGS
    include(test*".jl")
  end
end

end

@testset verbose = true "DF Calculations" begin

tests = ["df_hf", "df_uhf", "df_mcscf"]
for test in tests
  if runall || test in ARGS
    include(test*".jl")
  end
end

end

@testset verbose = true "Unit tests" begin

tests = ["unit_tests"]
for test in tests
  if runall || test in ARGS
    include(test*".jl")
  end
end

end
