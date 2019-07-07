__precompile__()

module VarianceComponentSelect

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
@reexport using PenaltyFunctions
@reexport using Plots 

import Base: size, length, +
export 
# mutable struct 
    VCModel, 
# operations 
    clamp_diagonal!,
    fixedeffects,
    kronaxpy!,
    nullprojection,  
    objvalue,  
# maximum lambda 
    maxlambda, 
# utilities function 
    checkfrobnorm!,
    matarray2mat,
    nvarcomps,
    plotsolpath,  
    rankvarcomps,   
    updateΩ!, update_arrays!,
# algorithm 
    mm_update_σ2!, mm_update_Σ!,
    vcselect, vcselect!, vcselectpath, vcselectpath!

"""
    VCModel 
    VCModel(Y, V)

Variance component model type. Stores the data and model parameters of a variance 
component model. 
"""
mutable struct VCModel{T <: Real} 
    # data
    Yobs        :: AbstractVecOrMat{T}
    Xobs        :: AbstractVecOrMat{T}
    Vobs        :: AbstractVector{Matrix{T}}
    Y           :: AbstractVecOrMat{T}       # projected response matrix 
    vecY        :: AbstractVector{T}         # vectorized projected response matrix 
    V           :: AbstractVector{Matrix{T}} # projected V
    # model parameters 
    β           :: AbstractVecOrMat{T}
    Σ           :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}}
    # covariance matrix and working arrays 
    Ω           :: AbstractMatrix{T}         # covariance matrix 
    Ωchol       :: Cholesky
    Ωinv        :: AbstractMatrix{T}         # inverse of covariance matrix 
    ΩinvY       :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
    R           :: AbstractVecOrMat{T}
    kron_ones_V :: AbstractVector{Matrix{T}}
    L           :: AbstractVecOrMat{T}
    Linv        :: AbstractVecOrMat{T}
    Mndxnd      :: AbstractMatrix{T}
    Mdxd        :: AbstractVecOrMat{T}
    Mnxd        :: AbstractVecOrMat{T}
    # covariance matrix in original dimension 
    Ωobs        :: AbstractMatrix{T}
end


"""
    VCModel(yobs, Xobs, Vobs, σ2)

Default constructor of [`VCModel`](@ref) type when `y` is vector. 
"""
function VCModel(
    yobs  :: AbstractVector{T},
    Xobs  :: AbstractVecOrMat{T},
    Vobs  :: AbstractVector{Matrix{T}},
    σ2 :: AbstractVector{T}
    ) where {T <: Real}

    # covariance matrix using original scale 

    y, V, = nullprojection(yobs, Xobs, Vobs)
    n = length(y)
    vecY = y 
    # 
    β = Vector{T}(undef, size(Xobs, 2)) # px1 
    # accumulate covariance matrix 
    Ω = zeros(T, n, n)
    for j in 1:length(σ2) 
        Ω .+= σ2[j] .* V[j]
    end 
    # allocate arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    ΩinvY = Ωinv * y
    R = reshape(ΩinvY, n, 1)
    kron_ones_V = similar(V)
    L = Vector{T}(undef, 1)
    Linv = Vector{T}(undef, 1)
    Mndxnd = Matrix{T}(undef, n, 0)
    Mdxd = Vector{T}(undef, 1)
    Mnxd = Vector{T}(undef, n) # nx1
    # allocate matrix in obs diemension
    Ωobs = Matrix{T}(undef, size(yobs, 1), size(yobs, 1))

    # 
    VCModel{T}(yobs, Xobs, Vobs, y, vecY, V,
            β, σ2, Ω, Ωchol, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωobs)
end 

"""
    VCModel(yobs, Vobs, σ2)

Construct [`VCModel`](@ref) from `y` and `V` where `y` is vector. `X` is treated empty. 
"""
function VCModel(
    yobs  :: AbstractVector{T},
    Vobs  :: AbstractVector{Matrix{T}},
    σ2 :: AbstractVector{T}
    ) where {T <: Real}

    Xobs = zeros(T, length(yobs), 0)
    VCModel(yobs, Xobs, Vobs, σ2)
end 

"""
    VCModel(Y, X, V, Σ)

Default constructor of [`VCModel`](@ref) type.

** components of V need to have frobenius norm 1 ** 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Xobs  :: AbstractVecOrMat{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}}
    ) where {T <: AbstractFloat}

    Y, V, = nullprojection(Yobs, Xobs, Vobs)
    n, d = size(Y, 1), size(Y, 2)
    vecY = vec(Y)
    nd = n * d
    p = size(Xobs, 2)
    # 
    β = Matrix{T}(undef, p, d)
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
    Ωobs = Matrix{T}(undef, length(Yobs), length(Yobs))

    # 
    VCModel{T}(Yobs, Xobs, Vobs, Y, vecY, V,
            β, Σ, Ω, Ωchol, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωobs)

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
    length(vcm)

Length `d` of response. 
"""
length(vcm::VCModel) = size(vcm.Y, 2)

"""
    ncovariates(vcm)

Number of fixed effects parameters `p` of a [`VCModel`](@ref). 
"""
ncovariates(vcm::VCModel) = size(vcm.Xobs, 2)

"""
    size(vcm)

Size `(n, d)` of response matrix of a [`VCModel`](@ref). 
"""
size(vcm::VCModel) = size(vcm.Y)

"""
    nmeanparams(vcm)

Number of mean parameters `p * d` of [`VCModel`](@ref).
"""
nmeanparams(vcm::VCModel) = length(vcm.β)

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
    updateΩobs!(vcm)

Update covariance matrix `Ωobs`. `Ωobs` has the same dimension as `Vobs`. 
"""
function updateΩobs!(vcm::VCModel)

    if isempty(vcm.Xobs) 
        vcm.Ωobs = vcm.Ω
    else
        fill!(vcm.Ωobs, 0)
        for k in 1:nvarcomps(vcm)
            kronaxpy!(vcm.Σ[k], vcm.Vobs[k], vcm.Ωobs)
        end
    end 
    vcm.Ωobs
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
