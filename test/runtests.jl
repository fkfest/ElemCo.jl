using Test

# choose what to test with `Pkg.test("ElemCo", test_args=["h2o","df_hf","quick"])`
# or `$ julia runtests.jl h2o df_hf quick`
# If no arguments are given, all quick tests are run.
# If "all" is given, all tests are run.
# If the name of a test set is given, all tests in that set are run.

runall = "all" in ARGS
runquick = length(ARGS) == 0 
if runall
  println("Running all tests")
elseif runquick
  println("Running quick tests")
else
  println("Running only $ARGS")
end

# quick tests
# [(testset, tests), ...]
# testset is the name of the test set
# tests is a list of test file names (without the .jl extension)
TESTS = [
("FCIDUMP", ["h2o", "h2o_st1", "n_st1", "h2o_cation", "h2o_anion_st1", "h2o_triplet", "2d_cc"]),
("CC", ["h2-"]),
("QV-CC", ["h2o_qv-ccd"]),
("DF", ["df_hf", "basis", "df_uhf", "df_mcscf"]),
("POS", ["pos_df_hf"]),
("SVD", ["svd_dc"]),
("Interface", ["h2o_matrop", "h2o_atomsbase"]),
("Unit-tests", ["unit_tests"])
]

# long tests
LONGTESTS = [
("Props", ["h2o_udcsd_prop"]),
("DMRG", ["h2o_dmrg"]),
("High-order CC", ["uccsdt", "ccsdt","svd_dc_ccsdt"]),
]

for (testset, tests) in TESTS
  doall = runall || runquick || testset in ARGS
  @testset verbose = true "$testset" begin
    for test in tests
      if doall || test in ARGS
        include(test*".jl")
      end
    end
  end
end

for (testset, tests) in LONGTESTS
  doall = runall || testset in ARGS
  @testset verbose = true "$testset" begin
    for test in tests
      if doall || test in ARGS
        include(test*".jl")
      end
    end
  end
end
