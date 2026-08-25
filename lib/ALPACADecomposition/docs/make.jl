using Documenter
using ALPACADecomposition

makedocs(;
  modules=[ALPACADecomposition],
  sitename="ALPACADecomposition.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", nothing) == "true",
    canonical="https://fkfest.github.io/ElemCo.jl/alpaca/",
    size_threshold=200 * 1024,
  ),
  pages=[
    "Home" => "index.md",
    "Theory" => "theory.md",
    "Tutorial" => "tutorial.md",
    "API Reference" => "api.md",
  ],
)
