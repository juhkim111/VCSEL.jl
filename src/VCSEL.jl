__precompile__()

module VCSEL

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
using Permutations
@reexport using PenaltyFunctions
@reexport using Plots 

import Base: size, length, +
export 
# struct 
    VCModel, VCintModel, 
# operations 
    clamp_diagonal!,
    kronaxpy!,
    nullprojection,  
    objvalue,  
    formΩ!,
# maximum lambda 
    findmaxλ, 
# utilities function 
    ncovariates, nvarcomps, nmeanparams, 
    plotsolpath, resetModel!, 
    rankvarcomps,   
    updateΩ!, update_arrays!,
# algorithm 
    mm_update_σ2!, mm_update_Σ!,
    vcselect, vcselect!, vcselectpath, vcselectpath!

"""
    VCModel(Y, X, G, Σ)
    VCModel(Y, X, G)
    VCModel(Y, G, Σ)
    VCModel(Y, G)

Variance component model type. Stores the data and model parameters of a variance 
component model. 


"""
struct VCModel{T <: Real} 
    # data
    Yobs        :: AbstractVecOrMat{T}
    Xobs        :: AbstractVecOrMat{T}
    Gobs        :: AbstractVector{Matrix{T}} 
    vecY        :: AbstractVector{T}         # vectorized projected response matrix 
    G           :: AbstractVector{Matrix{T}} # projected Gi including the last variance component 
    # model parameters 
    β           :: AbstractVecOrMat{T}
    Σ           :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}}
    # covariance matrix and working arrays  
    ΩcholL      :: LowerTriangular{T}        # cholesky factor 
    Ωinv        :: AbstractMatrix{T}         # inverse of covariance matrix 
    ΩinvY       :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
    R           :: AbstractVecOrMat{T}
    L           :: AbstractVecOrMat{T}
    Mndxnd      :: AbstractMatrix{T}
    kron_I_one  :: AbstractMatrix{T}
    Mdxd        :: AbstractVecOrMat{T}
    Mnxd        :: AbstractVecOrMat{T}
end


"""
    VCModel(Yobs, Xobs, Gobs, [Σ])

Default constructor of [`VCModel`](@ref) type.
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Xobs  :: AbstractVecOrMat{T},
    Gobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = 
                [Matrix{T}(I, size(Yobs, 2), size(Yobs, 2)) for i in 1:(length(Gobs)+1)]
    ) where {T <: AbstractFloat}

    # handle error 
    @assert (length(Gobs) + 1) == length(Σ) "length of vector of genotype matrices should be one less than that of variance components!\n"
    
    # projection 
    vecY, G, = nullprojection(Yobs, Xobs, Gobs) 
    n, d = size(G[1], 1), size(Yobs, 2)
    nd = n * d 
    p = size(Xobs, 2)
    # 
    β = Matrix{T}(undef, p, d)
    # #
    # wt = ones(length(Vobs))
    # accumulate covariance matrix 
    Ωinv = zeros(T, nd, nd)
    formΩ!(Ωinv, Σ, G)
    # pre-allocate working arrays 
    Ωchol = cholesky!(Symmetric(Ωinv))
    Ωinv[:] = inv(Ωchol) 
    ΩinvY = Ωinv * vecY  
    R = reshape(ΩinvY, n, d) # n x d 
    L = Matrix{T}(undef, d, d) # d x d 
    Mndxnd = Matrix{T}(undef, nd, nd) # nd x nd 
    kron_I_one = kron(I(d), ones(n)) # nd x d
    Mdxd = Matrix{T}(undef, d, d) # d x d 
    Mnxd = Matrix{T}(undef, n, d) # n x d 

    # 
    VCModel{T}(Yobs, Xobs, Gobs, vecY, G,
            β, Σ, Ωchol.L, Ωinv, ΩinvY,
            #β, Σ, Ω, Ωchol.L, Ωinv, ΩinvY, 
            R, L, Mndxnd, kron_I_one, Mdxd, Mnxd)

end 

"""
    VCModel(Yobs, Gobs, [Σ])

Construct [`VCModel`](@ref) from `Yobs` and `Gobs` where `Yobs` is matrix. 
`X` is treated empty. 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Gobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = [Matrix{T}(I, size(Yobs, 2), size(Yobs, 2)) for i in 1:(length(Gobs)+1)]
    ) where {T <: Real}

    # handle error 
    @assert (length(Gobs)+1) == length(Σ) "length of vector of genotype matrices should be one less than that of variance components!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, size(Yobs, 1), 0) 

    # call VCModel constructor 
    VCModel(Yobs, Xobs, Gobs, Σ)
end 

"""
    VCintModel(Y, X, V, Vint, Σ, Σint)
    VCintModel(Y, X, V, Vint)
    VCintModel(Y, V, Vint, Σ, Σint)
    VCintModel(Y, V, Vint)

Variance component interaction model type. Stores the data and model parameters of a variance 
component model. 
"""
struct VCintModel{T <: Real} 
    # data
    Yobs        :: AbstractVecOrMat{T}
    Xobs        :: AbstractVecOrMat{T}
    Vobs        :: AbstractVector{Matrix{T}}
    Vintobs     :: AbstractVector{Matrix{T}}
    Y           :: AbstractVecOrMat{T}       # projected response matrix 
    vecY        :: AbstractVector{T}         # vectorized projected response matrix 
    V           :: AbstractVector{Matrix{T}} # projected V
    Vint        :: AbstractVector{Matrix{T}}
    # model parameters 
    β           :: AbstractVecOrMat{T}
    Σ           :: Union{AbstractVector{T}}
    Σint        :: Union{AbstractVector{T}}
    # covariance matrix and working arrays 
    wt          :: AbstractVector{T}         # weight for standardization
    wt_int      :: AbstractVector{T}         # weight for standardization  
    Ω           :: AbstractMatrix{T}         # covariance matrix 
    ΩcholL      :: LowerTriangular{T}        # cholesky factor 
    Ωinv        :: AbstractMatrix{T}         # inverse of covariance matrix 
    ΩinvY       :: AbstractVector{T}         # Ωinv * Y OR Ωinv * vec(Y) 
    Mnxd        :: AbstractVecOrMat{T}
    # covariance matrix in original dimension 
    Ωest        :: AbstractMatrix{T}
end

"""
    VCintModel(yobs, Xobs, Vobs, Vintobs, [σ2, σ2int])

Default constructor of [`VCintModel`](@ref) type when `y` is vector. 
"""
function VCintModel(
    yobs    :: AbstractVecOrMat{T},
    Xobs    :: AbstractVecOrMat{T},
    Vobs    :: AbstractVector{Matrix{T}},
    Vintobs :: AbstractVector{Matrix{T}},
    σ2      :: AbstractVector{T} = ones(length(Vobs)),
    σ2int   :: AbstractVector{T} = ones(length(Vintobs))
    ) where {T <: Real}

    # handle error 
    @assert size(yobs, 2) == 1 "VCintModel can take only a vector!\n"

    # project onto null space 
    y, V, Vint, = nullprojection(yobs, Xobs, Vobs, Vintobs)
    n = size(y, 1)
    vecY = y 

    # initialize fixed effects parameter 
    β = Vector{T}(undef, size(Xobs, 2)) # px1 

    # initialize weight vector 
    wt = ones(length(Vobs))
    wt_int = ones(length(Vintobs))

    # overall covariance matrix 
    Ω = zeros(T, n, n)
    for j in 1:length(σ2int)
        axpy!(σ2[j], V[j], Ω) 
        axpy!(σ2int[j], Vint[j], Ω) 
    end
    axpy!(σ2[end], V[end], Ω)

    # allocate arrays 
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    ΩinvY = Ωinv * y
    Mnxd = Vector{T}(undef, n) # nx1
    # allocate matrix in obs diemension
    Ωest = Matrix{T}(undef, size(yobs, 1), size(yobs, 1))

    # 
    VCintModel{T}(yobs, Xobs, Vobs, Vintobs, y, vecY, V, Vint,
            β, σ2, σ2int, wt, wt_int, Ω, Ωchol.L, Ωinv, ΩinvY, 
            Mnxd, Ωest)
end 

"""
    VCintModel(yobs, Vobs, Vintobs, [σ2, σ2int])

Construct [`VCintModel`](@ref) from `y`, `V`, and `Vint` where `y` is vector. 
`X` is treated empty. 
"""
function VCintModel(
    yobs    :: AbstractVecOrMat{T},
    Vobs    :: AbstractVector{Matrix{T}},
    Vintobs :: AbstractVector{Matrix{T}},
    σ2      :: AbstractVector{T} = ones(length(Vobs)),
    σ2int   :: AbstractVector{T} = ones(length(Vintobs))
    ) where {T <: Real}

    # handle error 
    @assert length(Vobs) == length(σ2) "vector of covariance matrices and vector of variance components should have the same length!\n"
    @assert length(Vintobs) == length(σ2int) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, length(yobs), 0)

    # call VCintModel constructor 
    VCintModel(yobs, Xobs, Vobs, Vintobs, σ2, σ2int)
end 

"""
    length(vcm::VCModel)
    length(vcm::VCintModel)

Length `d` of response. 
"""
length(vcm::VCModel) = size(vcm.Yobs, 2)
length(vcm::VCintModel) = size(vcm.Y, 2)

"""
    ncovariates(vcm::VCModel)
    ncovariates(vcm::VCintModel)

Number of fixed effects parameters `p` of a [`VCModel`](@ref) or [`VCintModel`](@ref). 
"""
ncovariates(vcm::VCModel) = size(vcm.Xobs, 2)
ncovariates(vcm::VCintModel) = size(vcm.Xobs, 2)

"""
    size(vcm::VCModel)
    size(vcm::VCintModel)

Size `(n, d)` of response matrix of a [`VCModel`](@ref) or [`VCintModel`](@ref). 
"""
size(vcm::VCModel) = (size(vcm.G[1], 1), size(vcm.Yobs, 2)) # size(vcm.Y)
size(vcm::VCintModel) = size(vcm.Y)

"""
    nmeanparams(vcm::VCModel)
    nmeanparams(vcm::VCintModel)

Number of mean parameters `p * d` of [`VCModel`](@ref) or [`VCintModel`](@ref).
"""
nmeanparams(vcm::VCModel) = length(vcm.β)
nmeanparams(vcm::VCintModel) = length(vcm.β)

"""
    nvarcomps(vcm::VCModel)
    nvarcomps(vcm::VCintModel)

Number of variance components. 
"""
nvarcomps(vcm::VCModel) = length(vcm.Σ)
nvarcomps(vcm::VCintModel) = length(vcm.Σ) + length(vcm.Σint)

"""
    update_arrays!(vcm)

Update working arrays `Ωchol`, `Ωinv`, `ΩinvY`, `R`.
"""
function update_arrays!(vcm::VCModel)
    Ωchol = cholesky!(Symmetric(vcm.Ωinv))
    vcm.ΩcholL .= Ωchol.L
    vcm.Ωinv[:] = inv(Ωchol)
    mul!(vcm.ΩinvY, vcm.Ωinv, vcm.vecY)
    vcm.R[:] = reshape(vcm.ΩinvY, size(vcm))
    nothing 
end 

"""
    resetModel!(vcm::VCModel)

Reset [`VCModel`](@ref) with initial estimates `Σ`. If `Σ` is unspecified, 
it is set to a vector of ones or identity matrices based on its dimension.

# Example

```julia
vcm = VCModel(Y, X, G)
Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=50)
resetModel!(vcm)
```
"""
function resetModel!(
    vcm :: VCModel{T}
    ) where {T <: Real}
    d = length(vcm)
    resetModel!(vcm, 
        [Matrix{T}(I, d, d) for i in 1:nvarcomps(vcm)])
   
end 

"""
    resetModel!(vcm::VCModel, Σ)

Reset [`VCModel`](@ref) with initial estimates `Σ`.

# Example

```julia
d = size(Y, 2)
vcm = VCModel(Y, X, G)
Σ̂path, β̂path, = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=30)
Σ = [ones(d, d) for i in 1:nvarcomps(vcm)]
resetModel!(vcm, Σ)
```
"""
function resetModel!(
    vcm :: VCModel,
    Σ :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    (vcm.Σ)[:] = deepcopy(Σ)
    formΩ!(vcm.Ωinv, vcm.Σ, vcm.G)
    update_arrays!(vcm)
end 


include("vcselect.jl")
include("vcselect_interact.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalgops.jl")

end # module