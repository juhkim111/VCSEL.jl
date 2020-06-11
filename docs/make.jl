push!(LOAD_PATH, "../src/")
using Documenter, VCSEL

makedocs(
    doctest = false, 
    format = Documenter.HTML(),
    sitename = "VCSEL.jl",
    authors = "Juhyun Kim",
    clean = true, 
    pages = [
        "Home" => "index.md",
        "What is VCSEL?" => "VCSEL.md",
        "Manual" => Any[
            "man/VCModel.md",
            "man/VCintModel.md"
        ],
        # "Example: VCModel" => "man/VCModel_example.md",
        # "Example: VCintModel" => "man/VCintModel_example.md"
        "Examples" => Any[
            "man/VCModel_example.md",
            "man/VCintModel_example.md"
        ],
        #"API" => "man/api.md",
    ]
)

deploydocs(
    repo = "github.com/juhkim111/VCSEL.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)