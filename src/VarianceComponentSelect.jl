__precompile__()

module VarianceComponentSelect

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
@reexport using PenaltyFunctions
@reexport using Plots 

include("vcselect.jl")
include("vcselect_multivariate.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalg_operations.jl")

end # module
