using Documenter, ContinuousStatePopulationDynamics

makedocs(;
    modules = [ContinuousStatePopulationDynamics],
    warnonly = true,
    authors = "Simon Frost",
    sitename = "ContinuousStatePopulationDynamics.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/ContinuousStatePopulationDynamics.jl",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/ContinuousStatePopulationDynamics.jl.git",
)
