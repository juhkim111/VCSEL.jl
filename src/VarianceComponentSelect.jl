__precompile__()

module VarianceComponentSelect

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
@reexport using PenaltyFunctions
@reexport using Plots 

import Base: size, +
export 
# 
    VCModel, 
# operations 
    nullprojection, fixedeffects, kronaxpy!, clamp_diagonal!, objvalue,
# maximum lambda 
    maxlambda, 
# utilities function 
    plotsolpath, checkfrobnorm!, rankvarcomps, matarray2mat, nvarcomps,
# algorithm 
    vcselect, vcselectpath

"""
    VCModel 
    VCModel(Y, V)

Variance component model type. Stores the data and model parameters of a variance 
component model. 
"""
struct VCModel{T <: Real} 
    # data
    Y      :: AbstractVecOrMat{T}
    X      :: AbstractVecOrMat{T}
    V      :: AbstractVector{Matrix{T}}
    Ynew   :: AbstractVecOrMat{T}       # projected response 
    Vnew   :: AbstractVector{Matrix{T}} # projected V
    # model parsameters 
    Σ      :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}}
    # working arrays 
    Ω      :: AbstractMatrix{T}         # covariance matrix 
    Ωchol  :: Cholesky
    Ωinv   :: AbstractMatrix{T}         # inverse of covariance matrix 
    ΩinvY  :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
    tmpvec :: Vector{T}
    tmpmat :: Matrix{T}
end 

"""
    VCModel(y, X, V, σ2)

Default constructor of [`VCModel`](@ref) type when `y` is vector. 
"""
function VCModel(
    y  :: AbstractVector{T},
    X  :: AbstractMatrix{T},
    V  :: AbstractVector{Matrix{T}},
    σ2 :: AbstractVector{T}
    ) where T <: Real

    ynew, Vnew, = nullprojection(y, X, V)
    n = length(ynew)
    # accumulate covariance matrix 
    Ω = zeros(T, n, n)
    for j in 1:length(σ2) 
        Ω .+= σ2[j] .* Vnew[j]
    end 
    # allocate arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    ΩinvY = Ωinv * ynew
    tmpvec = similar(ynew)
    tmpmat = zeros(T, 0, 0)

    # 
    VCModel{T}(y, X, V, ynew, Vnew, σ2, Ω, Ωchol, Ωinv, ΩinvY, tmpvec, tmpmat)
end 

"""
    VCModel(y, V, σ2)

Construct [`VCModel`](@ref) from `y` and `V` where `y` is vector. `X` is treated empty. 
"""
function VCModel(
    y  :: AbstractVector{T},
    V  :: AbstractVector{Matrix{T}},
    σ2 :: AbstractVector{T}
    ) where T <: Real

    X = zeros(T, length(y), 0)
    VCModel(y, X, V, σ2)
end 

"""
    VCModel(Y, X, V, Σ)

Default constructor of [`VCModel`](@ref) type when `Y` is matrix. 
"""
function VCModel(
    Y  :: AbstractMatrix{T},
    X  :: AbstractMatrix{T},
    V  :: AbstractVector{Matrix{T}},
    Σ  :: AbstractVector{Matrix{T}}
    ) where T <: Real

    Ynew, Vnew, = nullprojection(Y, X, V)
    n, d = size(Ynew)
    nd = n * d
    # accumulate covariance matrix 
    Ω = zeros(T, nd, nd)
    for j in 1:length(Σ)
        kronaxpy!(Σ[j], V[j], Ω)
    end
    # allocate arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv[:] = inv(Ωchol) 
    vecY = vec(Y)
    ΩinvY = Ωinv * vecY  
    tmpvec = similar(vecY)
    tmpmat = similar(Ω)

    # 
    VCModel{T}(Y, X, V, Ynew, Vnew, Σ, Ω, Ωchol, Ωinv, ΩinvY, tmpvec, tmpmat)

end 

"""
    VCModel(Y, V, Σ)

Construct [`VCModel`](@ref) from `Y` and `V` where `Y` is matrix. `X` is treated empty. 
"""
function VCModel(
    Y  :: AbstractMatrix{T},
    V  :: AbstractVector{Matrix{T}},
    Σ  :: AbstractVector{Matrix{T}}
    ) where T <: Real

    X = zeros(T, size(Y, 1), 0) 
    VCModel{T}(Y, X, V, Σ)

end 

"""
    size(vcm)

size `(n, d)` of response matrix of a [`VCModel`](@ref). 
"""
size(vcm::VCModel) = size(vcm.Ynew)

"""
    nvarcomps(vcm)

Number of variance components. 
"""
nvarcomps(vcm::VCModel) = length(vcm.Σ)

# function updateΩ!

# end 






include("vcselect.jl")
include("vcselect_multivariate.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalg_operations.jl")

end # module
