using ReTestItems

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

# Limit BLAS threads per worker to avoid oversubscription when running in
# parallel (each worker would otherwise grab all cores for MKL).
const worker_init = nworkers > 1 ? quote
  using LinearAlgebra
  BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ $nworkers))
end : :()

if runall
  println("Running all ElemCo tests (including long-running ones); nworkers=$nworkers")
  # Everything except the known-broken orphan `pos_mp2` (tagged :broken).
  ReTestItems.runtests(@__DIR__; nworkers, worker_init_expr=worker_init,
                       name = r"^(?!pos_mp2$)")
else
  println("Running quick ElemCo tests (tag :quick); nworkers=$nworkers")
  ReTestItems.runtests(@__DIR__; nworkers, worker_init_expr=worker_init, tags=:quick)
end
