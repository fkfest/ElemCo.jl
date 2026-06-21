# Run ElemCo tests via the TestItems framework (`@testitem`), using ReTestItems
# as the runner so tests can optionally run in parallel across worker processes.
#
# The test-only deps (ReTestItems, TestItems, Test) live in [extras]/[targets]
# of Project.toml, so run through Pkg.test (which builds the test environment):
#   Pkg.test("ElemCo")                              # quick tests (items tagged :quick)
#   Pkg.test("ElemCo"; test_args=["all"])           # all tests (incl. :long), minus :broken
# (A bare `julia test/runtests.jl` against the package env won't see ReTestItems.)
#
# Choosing what to run: `test_args` (ARGS) select @testitems. `"all"` is the only
# reserved keyword; every other token is matched, case-insensitively, against each
# item's tags OR its name (an item is named after its file stem, e.g. `h2o`):
#   Pkg.test("ElemCo"; test_args=["df"])            # all :df items (old "DF" group)
#   Pkg.test("ElemCo"; test_args=["complex","h2o"]) # all :complex items + the h2o item
#   Pkg.test("ElemCo"; test_args=["long"])          # all :long items
#   Pkg.test("ElemCo"; test_args=["pos_mp2"])       # one item by name (even if :broken)
# Category tags in use: :fcidump :cc :eom :fci :ciphi :qvcc :df :region :system
# :pos :svd :interface :unit :complex :properties :dmrg :highorder, plus :quick / :long.
# With no tokens, the :quick items run. :broken items run only when named
# explicitly (or via the `broken` token); they otherwise stay VS-Code-visible only.
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

const nworkers = parse(Int, get(ENV, "ELEMCO_TEST_NWORKERS", "0"))

# --- Test selection ---------------------------------------------------------
# Parse ARGS once (case-insensitively). `"all"` is the only reserved keyword
# (run everything except :broken); the remaining tokens are tag/name selectors.
const args = lowercase.(ARGS)
const runall = "all" in args
const selectors = filter(!=("all"), args)

# A selector token matches an item by one of its tags or by its (lowercased) name.
selector_matches(ti) = any(s -> Symbol(s) in ti.tags || s == lowercase(ti.name), selectors)

# Shared predicate for both runners; their filter argument exposes `.name`/`.tags`.
function selector(ti)
  if :broken in ti.tags
    # :broken items run only when named explicitly or via the `broken` token;
    # default / all / plain tag selection skip them (still visible in the Test Explorer).
    return ("broken" in selectors) || (lowercase(ti.name) in selectors)
  end
  runall             && return true                # everything except :broken
  isempty(selectors) && return :quick in ti.tags   # default: quick
  return selector_matches(ti)                      # tag- or name-selected
end

# Human-readable description of the current selection, for the run banner.
const selection_msg = runall ? "all tests (incl. long-running), except :broken" :
                      isempty(selectors) ? "quick tests (tag :quick)" :
                      "selected tests (tags/names: $(join(selectors, ", ")))"

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
# branch and the TestItemRunner test-dep and always use ReTestItems again.
# Note the `-` in `v"1.13-"`: it makes prereleases (e.g. 1.13.0-rc1) compare as
# >= 1.13, so they take the fallback. Plain `v"1.13"` would wrongly treat the rc
# as < 1.13 (prereleases sort before their release) and pick the broken runner.
const use_retestitems = VERSION < v"1.13-"

if use_retestitems
  using ReTestItems
  # Limit BLAS threads per worker to avoid oversubscription when running in
  # parallel (each worker would otherwise grab all cores for MKL).
  const worker_init = nworkers > 1 ? quote
    using LinearAlgebra
    BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ $nworkers))
  end : :()
  println("Running $selection_msg; nworkers=$nworkers")
  ReTestItems.runtests(selector, @__DIR__; nworkers, worker_init_expr=worker_init)
else
  using TestItemRunner
  nworkers != 0 && @warn "ELEMCO_TEST_NWORKERS is ignored on Julia >= 1.13: the TestItemRunner fallback runs single-process."
  # Same discovery scope as the ReTestItems path (`@__DIR__`), so nested
  # `lib/ALPACADecomposition` and `.claude/worktrees/*` items are not picked up.
  # Filtering uses the same `selector` predicate; for TestItemRunner it sees a
  # NamedTuple whose `name`/`tags` fields hold the item's `@testitem` name and tags.
  println("[Julia $VERSION] Using TestItemRunner fallback (ReTestItems is broken on 1.13).")
  println("Running $selection_msg; single-process.")
  TestItemRunner.run_tests(@__DIR__; filter=selector)
end
