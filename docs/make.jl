using Documenter, ContinuousStatePopulationDynamics

# Copy the rendered vignettes (vignettes/NN_name/NN_name.md and NN_name_files/) into
# docs/src/tutorials/ at build time, so the docs serve them without a second copy in git.
# Quarto's table of contents is dropped: Documenter shows its own.
function sync_tutorials()
    vig = joinpath(@__DIR__, "..", "vignettes")
    out = joinpath(@__DIR__, "src", "tutorials")
    rm(out; recursive = true, force = true)
    mkpath(out)
    pages = Pair{String,String}[]
    for d in sort(filter(d -> occursin(r"^\d+_", d) && isfile(joinpath(vig, d, d * ".md")), readdir(vig)))
        text = read(joinpath(vig, d, d * ".md"), String)
        text = replace(text, r"\A(# [^\n]*\n)(?:[^\n#-][^\n]*\n)?\s*(?:[ \t]*- \[[^\n]*\n(?:[ \t]+[^\n]*\n)*)+"s => s"\1\n")
        write(joinpath(out, d * ".md"), text)
        files = joinpath(vig, d, d * "_files")
        isdir(files) && cp(files, joinpath(out, d * "_files"))
        m = match(r"^# (.+)$"m, text)
        push!(pages, (m === nothing ? d : m.captures[1]) => "tutorials/$d.md")
    end
    return pages
end

tutorials = sync_tutorials()

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
        "Tutorials" => tutorials,
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/ContinuousStatePopulationDynamics.jl.git",
    push_preview = true,
)
