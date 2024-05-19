push!(LOAD_PATH,"../src/")
using Documenter, ElemCo, Changelog

DocMeta.setdocmeta!(ElemCo, :DocTestSetup, :(using ElemCo); recursive=true)

cp(joinpath(@__DIR__,"equations","equations.pdf"),joinpath(@__DIR__,"src","assets","equations.pdf"), force=true)

# Generate a Documenter-friendly changelog from CHANGELOG.md
Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src", "release-notes.md");
    repo = "fkfest/ElemCo.jl",
)

makedocs(
  modules = [ElemCo],
  format = Documenter.HTML(
    # Use clean URLs, unless built as a "local" build
    prettyurls = !("local" in ARGS),
    assets = ["assets/favicon.ico"],
  ),
  sitename="ElemCo.jl documentation",
  pages = [
    "Home" => "index.md",
    "Manual" => [
      "Running calculations" => "elemco.md",
      "Options" => "options.md" 
      ],
    "Internals" => [
      "basisset.md",
      "bohf.md",
      "cc.md",
      "ccdriver.md",
      "cctools.md",
      "constants.md",
      "decomptools.md",
      "dfcc.md",
      "dfdump.md",
      "dfhf.md",
      "dfmcscf.md",
      "dftools.md",
      "diis.md",
      "dump.md",
      "dumptools.md",
      "elements.md",
      "ecinfos.md",
      "ecmethod.md",
      "fockfactory.md",
      "integrals.md",
      "mio.md",
      "mnpy.md",
      "msystem.md",
      "orbtools.md",
      "tensortools.md",
      "utils.md"
    ],
    "release-notes.md", 
  ],
  checkdocs=:exports)

deploydocs(
    repo = "github.com/fkfest/ElemCo.jl-devel.git",
)
