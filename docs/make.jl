#push!(LOAD_PATH,"../src/")
using Documenter, VCSEL

makedocs(
    format = Documenter.HTML(),
    sitename = "VCSEL.jl",
    authors = "Juhyun Kim",
    pages = [
        "Home" => "index.md",
        #"Examples" => , 
        "API" => "man/api.md",
    ]
)

deploydocs(
    repo = "github.com/juhkim111/VCSEL.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)