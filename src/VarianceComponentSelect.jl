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
    plotsolpath, checkfrobnorm!, rankvarcomps, matarray2mat, nvarcomps, updateΩ!, 
    update_arrays!,
# algorithm 
    vcselect, vcselect!, vcselectpath

"""
    VCModel 
    VCModel(Y, V)

Variance component model type. Stores the data and model parameters of a variance 
component model. 
"""
# struct VCModel{T <: Real} 
#     # data
#     Y      :: Vector{T}
#     X      :: AbstractVecOrMat{T}
#     V      :: AbstractVector{Matrix{T}}
#     Ynew   :: AbstractVecOrMat{T}       # projected response 
#     Vnew   :: AbstractVector{Matrix{T}} # projected V
#     # model parsameters 
#     Σ      :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}}
#     # working arrays 
#     Ω      :: AbstractMatrix{T}         # covariance matrix 
#     Ωchol  :: Cholesky
#     Ωinv        :: AbstractMatrix{T}         # inverse of covariance matrix 
#     ΩinvY       :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
#     tmpvec      :: Vector{T}
# end

mutable struct VCModel{T <: Real} 
    # data
    Yobs       :: AbstractMatrix{T}
    Xobs       :: AbstractVecOrMat{T}
    Vobs       :: AbstractVector{Matrix{T}}
    Y          :: AbstractVecOrMat{T}       # projected response 
    vecY       :: AbstractVector{T}
    V          :: AbstractVector{Matrix{T}} # projected V
    # model parameters 
    Σ       :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}}
    # working arrays 
    Ω       :: AbstractMatrix{T}         # covariance matrix 
    Ωchol   :: Cholesky
    Ωinv        :: AbstractMatrix{T}         # inverse of covariance matrix 
    ΩinvY       :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
    R           :: AbstractMatrix{T}
    kron_ones_V :: AbstractVector{Matrix{T}}
    L           :: AbstractMatrix{T}
    Linv        :: AbstractMatrix{T}
    Mndxnd      :: AbstractMatrix{T}
    Mdxd        :: AbstractMatrix{T}
    Mnxd        :: AbstractMatrix{T}
    #storage     :: LinearAlgebra.Eigen 
end


# """
#     VCModel(y, X, V, σ2)

# Default constructor of [`VCModel`](@ref) type when `y` is vector. 
# """
# function VCModel(
#     y  :: AbstractVector{T},
#     X  :: AbstractMatrix{T},
#     V  :: AbstractVector{Matrix{T}},
#     σ2 :: AbstractVector{T}
#     ) where T <: Real

#     ynew, Vnew, = nullprojection(y, X, V)
#     n = length(ynew)
#     # accumulate covariance matrix 
#     Ω = zeros(T, n, n)
#     for j in 1:length(σ2) 
#         Ω .+= σ2[j] .* Vnew[j]
#     end 
#     # allocate arrays 
#     Ωchol = cholesky!(Symmetric(Ω))
#     Ωinv = inv(Ωchol) 
#     ΩinvY = Ωinv * ynew
#     tmpvec = similar(ynew)
#     tmpmat = zeros(T, 0, 0)

#     # 
#     VCModel{T}(y, X, V, ynew, Vnew, σ2, Ω, Ωchol, Ωinv, ΩinvY, tmpvec, tmpmat)
# end 

# """
#     VCModel(y, V, σ2)

# Construct [`VCModel`](@ref) from `y` and `V` where `y` is vector. `X` is treated empty. 
# """
# function VCModel(
#     y  :: AbstractVector{T},
#     V  :: AbstractVector{Matrix{T}},
#     σ2 :: AbstractVector{T}
#     ) where T <: Real

#     X = zeros(T, length(y), 0)
#     VCModel(y, X, V, σ2)
# end 

"""
    VCModel(Y, X, V, Σ)

Default constructor of [`VCModel`](@ref) type when `Y` is matrix.

** components of V need to have frobenius norm 1 ** 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Xobs  :: AbstractMatrix{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}}
    ) where {T <: AbstractFloat}

    Y, V, = nullprojection(Yobs, Xobs, Vobs)
    n, d = size(Y)
    vecY = vec(Y)
    nd = n * d
    # accumulate covariance matrix 
    Ω = zeros(T, nd, nd)
    for j in 1:length(Σ)
        kronaxpy!(Σ[j], V[j], Ω)
    end
    # pre-allocate working arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    ΩinvY = Ωinv * vecY  
    R = reshape(ΩinvY, n, d)
    kron_ones_V = similar(V)
    L = Matrix{T}(undef, d, d)
    Linv = Matrix{T}(undef, d, d)
    Mndxnd = Matrix{T}(undef, nd, nd)
    Mdxd = Matrix{T}(undef, d, d)
    Mnxd = Matrix{T}(undef, n, d)

    # 
    VCModel{T}(Yobs, Xobs, Vobs, Y, vecY, V,
            Σ, Ω, Ωchol, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd)

end 

"""
    VCModel(Y, V, Σ)

Construct [`VCModel`](@ref) from `Y` and `V` where `Y` is matrix. `X` is treated empty. 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    Xobs = zeros(T, size(Yobs, 1), 0) 
    VCModel(Yobs, Xobs, Vobs, Σ)

end 

"""
    size(vcm)

size `(n, d)` of response matrix of a [`VCModel`](@ref). 
"""
size(vcm::VCModel) = size(vcm.Y)

"""
    nvarcomps(vcm)

Number of variance components. 
"""
nvarcomps(vcm::VCModel) = length(vcm.Σ)

"""
    updateΩ!(vcm)

Update covariance matrix `Ω`.
"""
function updateΩ!(vcm::VCModel)
    fill!(vcm.Ω, 0)
    for k in 1:nvarcomps(vcm)
        kronaxpy!(vcm.Σ[k], vcm.V[k], vcm.Ω)
    end
    vcm.Ω
end

"""
    update_arrays!(vcm)

Update working arrays `Ωchol`, `Ωinv`, `ΩinvY`, `R`.
"""
function update_arrays!(vcm::VCModel)
    vcm.Ωchol = cholesky!(Symmetric(vcm.Ω))
    vcm.Ωinv[:] = inv(vcm.Ωchol)
    mul!(vcm.ΩinvY, vcm.Ωinv, vcm.vecY)
    vcm.R = reshape(vcm.ΩinvY, size(vcm))
    nothing 
end 






include("vcselect.jl")
include("vcselect_multivariate.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalg_operations.jl")

end # module
