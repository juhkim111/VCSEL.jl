__precompile__()

module VarianceComponentSelect

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
@reexport using PenaltyFunctions
@reexport using Plots 

import Base: size, length, +
export 
# struct 
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
    matarray2mat,
    nvarcomps,
    plotsolpath, resetVCModel!, 
    rankvarcomps,   
    updateΩ!, update_arrays!,
# algorithm 
    mm_update_σ2!, mm_update_Σ!,
    vcselect, vcselect!, vcselectpath, vcselectpath!

"""
    VCModel 
    VCModel(Y, X, V, Σ)
    VCModel(Y, X, V)
    VCModel(Y, V, Σ)
    VCModel(Y, V)

Variance component model type. Stores the data and model parameters of a variance 
component model. 
"""
struct VCModel{T <: Real} 
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
    wt          :: AbstractVector{T}         # weight for standardization 
    Ω           :: AbstractMatrix{T}         # covariance matrix 
    ΩcholL      :: LowerTriangular{T}        # cholesky factor 
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
    VCModel(yobs, Xobs, Vobs, [σ2])

Default constructor of [`VCModel`](@ref) type when `y` is vector. 
"""
function VCModel(
    yobs  :: AbstractVector{T},
    Xobs  :: AbstractVecOrMat{T},
    Vobs  :: AbstractVector{Matrix{T}},
    σ2    :: AbstractVector{T} = ones(length(Vobs))
    ) where {T <: Real}

    # handle error 
    @assert length(Vobs) == length(σ2) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # covariance matrix using original scale 
    y, V, = nullprojection(yobs, Xobs, Vobs)
    n = length(y)
    vecY = y 
    # 
    β = Vector{T}(undef, size(Xobs, 2)) # px1 
    # accumulate covariance matrix 
    wt = ones(length(Vobs))
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
            β, σ2, wt, Ω, Ωchol.L, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωobs)
end 


"""
    VCModel(yobs, Vobs, [σ2])

Construct [`VCModel`](@ref) from `y` and `V` where `y` is vector. `X` is treated empty. 
"""
function VCModel(
    yobs  :: AbstractVector{T},
    Vobs  :: AbstractVector{Matrix{T}},
    σ2    :: AbstractVector{T} = ones(length(Vobs))
    ) where {T <: Real}

    # handle error 
    @assert length(Vobs) == length(σ2) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, length(yobs), 0)

    # call VCModel constructor 
    VCModel(yobs, Xobs, Vobs, σ2)
end 

"""
    VCModel(Yobs, Xobs, Vobs, [Σ])

Default constructor of [`VCModel`](@ref) type.
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Xobs  :: AbstractVecOrMat{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = [Matrix(one(T)*I, 
                                        size(Yobs, 2), size(Yobs, 2)) for i in 1:length(Vobs)]
    ) where {T <: AbstractFloat}

    # handle error 
    @assert length(Vobs) == length(Σ) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # projection 
    Y, V, = nullprojection(Yobs, Xobs, Vobs)
    n, d = size(Y, 1), size(Y, 2)
    vecY = vec(Y)
    nd = n * d
    p = size(Xobs, 2)
    # 
    β = Matrix{T}(undef, p, d)
    #
    wt = ones(length(Vobs))
    # accumulate covariance matrix 
    Ω = zeros(T, nd, nd)
    for j in 1:length(Σ)
        kronaxpy!(Σ[j], V[j], Ω)
    end
    # pre-allocate working arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    ΩinvY = Ωinv * vecY  
    R = reshape(ΩinvY, n, d) # n x d 
    kron_ones_V = similar(V)
    kron_I_one = kron(Matrix(I, d, d), ones(n)) # dn x d
    ones_d = ones(d, d)
    for i in 1:length(V)
        kron_ones_V[i] = kron(ones_d, V[i])
    end 
    L = Matrix{T}(undef, d, d)
    Linv = Matrix{T}(undef, d, d)
    Mndxnd = Matrix{T}(undef, nd, nd)
    Mdxd = Matrix{T}(undef, d, d) # d x d 
    Mnxd = Matrix{T}(undef, n, d)
    # 
    Ωobs = Matrix{T}(undef, length(Yobs), length(Yobs))

    # 
    VCModel{T}(Yobs, Xobs, Vobs, Y, vecY, V,
            β, Σ, wt, Ω, Ωchol.L, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωobs)

end 

"""
    VCModel(Yobs, Vobs, [Σ])

Construct [`VCModel`](@ref) from `Y` and `V` where `Y` is matrix. `X` is treated empty. 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = fill(ones(T, size(Y, 2), size(Y, 2)), length(V)),
    ) where {T <: Real}

    # handle error 
    @assert length(Vobs) == length(Σ) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, size(Yobs, 1), 0) 

    # call VCModel constructor 
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
        kronaxpy!(vcm.wt[k] .* vcm.Σ[k], vcm.V[k], vcm.Ω)
    end
    vcm.Ω
end

"""
    updateΩobs!(vcm)

Update covariance matrix `Ωobs`. `Ωobs` has the same dimension as `Vobs`. 
"""
function updateΩobs!(vcm::VCModel)

    if isempty(vcm.Xobs) 
        vcm.Ωobs .= vcm.Ω
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
    # vcm.Ωchol = cholesky!(Symmetric(vcm.Ω))
    # vcm.Ωinv[:] = inv(vcm.Ωchol)
    Ωchol = cholesky!(Symmetric(vcm.Ω))
    vcm.ΩcholL .= Ωchol.L
    vcm.Ωinv[:] = inv(Ωchol)
    mul!(vcm.ΩinvY, vcm.Ωinv, vcm.vecY)
    vcm.R .= reshape(vcm.ΩinvY, size(vcm))
    nothing 
end 

"""
    resetVCModel!(vcm, Σ)

Reset [`VCModel`](@ref) with initial estimates `Σ`.
"""
function resetVCModel!(
    vcm :: VCModel,
    Σ :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}} 
    ) where {T <: Real}

    vcm.Σ .= Σ
    updateΩ!(vcm)
    updateΩobs!(vcm)
    update_arrays!(vcm)
    vcm.R .= reshape(vcm.ΩinvY, size(vcm))
end 

"""
    resetVCModel!(vcm, Σ)

Reset [`VCModel`](@ref) with initial estimates `Σ`. If `Σ` is unspecified, it is set to 
a vector of ones or all-one matrices based on its dimension.
"""
function resetVCModel!(
    vcm :: VCModel
    ) 
    d = length(vcm)
    if d == 1
        fill!(vcm.Σ, 1)
        #vcm.Σ = ones(eltype(vcm.Σ), nvarcomps(vcm))
    else 
        fill!(vcm.Σ, ones(eltype(vcm.Σ[1]), d, d))
        #vcm.Σ = fill(ones(eltype(vcm.Σ[1]), length(vcm), length(vcm)), nvarcomps(vcm))
    end 
    updateΩ!(vcm)
    # allocate arrays 
    updateΩobs!(vcm)
    update_arrays!(vcm)
    vcm.R .= reshape(vcm.ΩinvY, size(vcm))
end 

include("vcselect.jl")
include("vcselect_multivariate.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalg_operations.jl")

end # module
