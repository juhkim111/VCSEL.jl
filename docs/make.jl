push!(LOAD_PATH,"../src/")
using Documenter, VarianceComponentSelect

makedocs(
    sitename = "VarianceComponentSelect.jl",
    authors = "Juhyun Kim",
    pages = [
        "index.md",
        "api.md"
    ]
)

deploydocs(
    repo = "github.com/juhkim111/VarianceComponentSelect.jl.git"
)