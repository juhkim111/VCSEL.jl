__precompile__()

module VarianceComponentSelect

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
@reexport using PenaltyFunctions
@reexport using Plots 

export vcselect, vcselectpath, maxlambda, nullprojection, 
        betaestimate, checkfrobnorm!, rankvarcomps, plotsolpath

include("vcselect.jl")
include("maxlambda.jl")
include("utilities.jl")

end # module
