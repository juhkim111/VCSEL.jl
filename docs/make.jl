using Documenter, VCSEL

makedocs(
    doctest = false, 
    format = Documenter.HTML(),
    sitename = "VCSEL.jl",
    authors = "Juhyun Kim",
    clean = true, 
    pages = [
        "Home"           => "index.md",
        "What is VCSEL?" => "man/VCSEL.md",
        "VCModel"        => "man/VCModel.md",
        "VCintModel"     => "man/VCintModel.md",
        "Examples"       => Any[
            "man/VCModel_example.md",
            "man/VCintModel_example.md"
        ],
        "API" => "man/api.md"
    ]
)

deploydocs(
    repo = "github.com/juhkim111/VCSEL.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)