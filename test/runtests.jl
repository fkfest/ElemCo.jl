# Run ElemCo tests via the TestItems framework (`@testitem`), using ReTestItems
# as the runner so tests can optionally run in parallel across worker processes.
#
# The test-only deps (ReTestItems, TestItems, Test) live in [extras]/[targets]
# of Project.toml, so run through Pkg.test (which builds the test environment):
#   Pkg.test("ElemCo")                          # quick tests (items tagged :quick)
#   Pkg.test("ElemCo"; test_args=["all"])        # all tests (incl. :long), minus :broken
# (A bare `julia test/runtests.jl` against the package env won't see ReTestItems.)
#
# Parallelism (off by default to match the historical single-process behaviour):
#   ELEMCO_TEST_NWORKERS=4 julia -e 'using Pkg; Pkg.test("ElemCo")'
#     0 (default) -> run sequentially in this process (ElemCo loaded once)
#     1           -> run sequentially in one fresh worker process
#     N>1         -> run @testitems in parallel across N worker processes
#   Each worker loads ElemCo once, so workers trade extra startup/memory for
#   wall-clock speedup. ElemCo uses MKL; to avoid BLAS oversubscription when
#   N is large, also lower the BLAS threads, e.g. set ELEMCO_TEST_NWORKERS and
#   pass `worker_init_expr` below, or run with `MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`.
#
# In VS Code, discovery/running of @testitems is handled by the Julia extension's
# own test process(es) and does not go through this file. The number of those
# processes is controlled by the `julia.numTestProcesses` setting.
#
# NOTE: discovery is scoped to this `test/` directory (via @__DIR__), not the
# package root, so the nested `lib/ALPACADecomposition` test items and any
# `.claude/worktrees/*` checkouts are not picked up. Each @testitem imports
# `using ElemCo` itself.

const runall = "all" in ARGS
const nworkers = parse(Int, get(ENV, "ELEMCO_TEST_NWORKERS", "0"))

# Never run items tagged :broken. They stay discoverable in the Test Explorer.
notbroken(ti) = !(:broken in ti.tags)

# --- Runner selection -------------------------------------------------------
# ReTestItems is our normal runner. It is broken on Julia >= 1.13 because
# `Test.TESTSET_PRINT_ENABLE` became a `ScopedValue{Bool}` there, while
# ReTestItems still does `TESTSET_PRINT_ENABLE[] = ...` (MethodError: no method
# matching setindex!(::ScopedValue{Bool}, ::Bool)). Until the upstream fix is
# released (https://github.com/JuliaTesting/ReTestItems.jl/pull/235) we fall back
# to TestItemRunner on 1.13+, which drives the same `@testitem`s but runs them
# single-process (no worker parallelism).
#
# TODO(ReTestItems-1.13): once a fixed ReTestItems is registered, drop this
# branch and the TestItemRunner test-dep and always use ReTestItems again. See
# tasks/lessons.md.
# Note the `-` in `v"1.13-"`: it makes prereleases (e.g. 1.13.0-rc1) compare as
# >= 1.13, so they take the fallback. Plain `v"1.13"` would wrongly treat the rc
# as < 1.13 (prereleases sort before their release) and pick the broken runner.
const use_retestitems = VERSION < v"1.13-"

if use_retestitems
  using ReTestItems
  # Limit BLAS threads per worker to avoid oversubscription when running in
  # parallel (each worker would otherwise grab all cores for MKL).
  worker_init = nworkers > 1 ? quote
    using LinearAlgebra
    BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ $nworkers))
  end : :()
  if runall
    println("Running all ElemCo tests (including long-running ones); nworkers=$nworkers")
    ReTestItems.runtests(notbroken, @__DIR__; nworkers, worker_init_expr=worker_init)
  else
    println("Running quick ElemCo tests (tag :quick); nworkers=$nworkers")
    ReTestItems.runtests(notbroken, @__DIR__; nworkers, worker_init_expr=worker_init, tags=:quick)
  end
else
  using TestItemRunner
  nworkers != 0 && @warn "ELEMCO_TEST_NWORKERS is ignored on Julia >= 1.13: the TestItemRunner fallback runs single-process."
  # Same discovery scope as the ReTestItems path (`@__DIR__`), so nested
  # `lib/ALPACADecomposition` and `.claude/worktrees/*` items are not picked up.
  # Filtering is done via the `filter` function (no `tags=` kwarg here); the
  # filter sees a NamedTuple whose `tags` field holds the item's `@testitem` tags.
  selector = runall ? notbroken : ti -> notbroken(ti) && :quick in ti.tags
  println("[Julia $VERSION] Using TestItemRunner fallback (ReTestItems is broken on 1.13).")
  println(runall ? "Running all ElemCo tests (including long-running ones); single-process." :
                   "Running quick ElemCo tests (tag :quick); single-process.")
  TestItemRunner.run_tests(@__DIR__; filter=selector)
end
