__precompile__()

module VCSEL

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
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
# maximum lambda 
    maxlambda, 
# utilities function 
    ncovariates, nvarcomps, nmeanparams, 
    plotsolpath, resetModel!, 
    rankvarcomps,   
    updateΩ!, update_arrays!,
# algorithm 
    mm_update_σ2!, mm_update_Σ!,
    vcselect, vcselect!, vcselectpath, vcselectpath!

"""
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
    Ωest        :: AbstractMatrix{T}
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

    # project onto null space 
    y, V, = nullprojection(yobs, Xobs, Vobs)
    n = length(y)
    vecY = y 

    # initialize fixed effects parameter 
    β = Vector{T}(undef, size(Xobs, 2)) # px1 

    wt = ones(length(Vobs))

    # overall covariance matrix 
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
    Ωest = Matrix{T}(undef, size(yobs, 1), size(yobs, 1))

    # 
    VCModel{T}(yobs, Xobs, Vobs, y, vecY, V,
            β, σ2, wt, Ω, Ωchol.L, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωest)
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
    Ωest = Matrix{T}(undef, length(Yobs), length(Yobs))

    # 
    VCModel{T}(Yobs, Xobs, Vobs, Y, vecY, V,
            β, Σ, wt, Ω, Ωchol.L, Ωinv, ΩinvY, 
            R, kron_ones_V, L, Linv, Mndxnd, Mdxd, Mnxd,
            Ωest)

end 

"""
    VCModel(Yobs, Vobs, [Σ])

Construct [`VCModel`](@ref) from `Y` and `V` where `Y` is matrix. `X` is treated empty. 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Vobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = [Matrix(one(T)*I, 
            size(Yobs, 2), size(Yobs, 2)) for i in 1:length(Vobs)]
    ) where {T <: Real}

    # handle error 
    @assert length(Vobs) == length(Σ) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, size(Yobs, 1), 0) 

    # call VCModel constructor 
    VCModel(Yobs, Xobs, Vobs, Σ)
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
length(vcm::VCModel) = size(vcm.Y, 2)
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
size(vcm::VCModel) = size(vcm.Y)
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
    ngroups(vcm::VCModel)
    ngroups(vcm::VCintModel)

Number of groups, `m`, for [`VCModel`](@ref) or [`VCintModel`](@ref).
"""
ngroups(vcm::VCModel) = nvarcomps(vcm) - 1
ngroups(vcm::VCintModel) = length(vcm.Σ) - 1

"""
    updateΩ!(vcm::VCModel)

Update covariance matrix `Ω` for [`VCModel`](@ref).
"""
function updateΩ!(vcm::VCModel)
    fill!(vcm.Ω, 0)
    for k in 1:nvarcomps(vcm)
        kronaxpy!(vcm.wt[k] .* vcm.Σ[k], vcm.V[k], vcm.Ω)
    end
    vcm.Ω
end

"""
    updateΩ!(vcm::VCintModel)

Update covariance matrix `Ω` for [`VCintModel`](@ref).
"""
function updateΩ!(vcm::VCintModel)
    fill!(vcm.Ω, 0)
    for k in 1:ngroups(vcm)
        kronaxpy!(vcm.wt[k] .* vcm.Σ[k], vcm.V[k], vcm.Ω)
        kronaxpy!(vcm.wt_int[k] .* vcm.Σint[k], vcm.Vint[k], vcm.Ω)
    end
    kronaxpy!(vcm.wt[end] .* vcm.Σ[end], vcm.V[end], vcm.Ω)
    vcm.Ω
end

"""
    updateΩest!(vcm::VCModel)

Update covariance matrix `Ωest` for [`VCModel`](@ref). `Ωest` has the same dimension as `Vobs`. 
"""
function updateΩest!(vcm::VCModel)

    if isempty(vcm.Xobs) 
        vcm.Ωest .= vcm.Ω
    else
        fill!(vcm.Ωest, 0)
        for k in 1:nvarcomps(vcm)
            kronaxpy!(vcm.Σ[k], vcm.Vobs[k], vcm.Ωest)
        end
    end 
    vcm.Ωest
end

"""
    updateΩest!(vcm::VCintModel)

Update covariance matrix `Ωest` for [`VCintModel`](@ref). `Ωest` has the same dimension as `Vobs`. 
"""
function updateΩest!(vcm::VCintModel)

    if isempty(vcm.Xobs) 
        vcm.Ωest .= vcm.Ω
    else
        fill!(vcm.Ωest, 0)
        for k in 1:ngroups(vcm)
            kronaxpy!(vcm.Σ[k], vcm.Vobs[k], vcm.Ωest)
            kronaxpy!(vcm.Σint[k], vcm.Vintobs[k], vcm.Ωest)
        end
        kronaxpy!(vcm.Σ[end], vcm.Vobs[end], vcm.Ωest)
    end 
    vcm.Ωest
end

"""
    update_arrays!(vcm)

Update working arrays `Ωchol`, `Ωinv`, `ΩinvY`, `R`.
"""
function update_arrays!(vcm::Union{VCModel, VCintModel})
    # vcm.Ωchol = cholesky!(Symmetric(vcm.Ω))
    # vcm.Ωinv[:] = inv(vcm.Ωchol)
    Ωchol = cholesky!(Symmetric(vcm.Ω))
    vcm.ΩcholL .= Ωchol.L
    vcm.Ωinv[:] = inv(Ωchol)
    mul!(vcm.ΩinvY, vcm.Ωinv, vcm.vecY)
    if typeof(vcm) <: VCModel
        vcm.R .= reshape(vcm.ΩinvY, size(vcm))
    end 
    nothing 
end 

"""
    resetModel!(vcm::VCModel)

Reset [`VCModel`](@ref) with initial estimates `Σ`. If `Σ` is unspecified, 
it is set to a vector of ones or identity matrices based on its dimension.

# Example

```julia
vcm = VCModel(Y, X, V)
Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=50)
resetModel!(vcm)
```
"""
function resetModel!(
    vcm :: VCModel
    ) 
    d = length(vcm)
    if typeof(vcm.Σ[1]) <: Matrix 
        resetModel!(vcm, 
                [Matrix(one(eltype(vcm.Σ[1]))*I, d, d) for i in eachindex(vcm.Σ)])
    else 
        resetModel!(vcm,
                ones(eltype(vcm.Σ[1]), nvarcomps(vcm)))
    end 
end 

"""
    resetModel!(vcm::VCModel, Σ)

Reset [`VCModel`](@ref) with initial estimates `Σ`.

# Example

```julia
vcm = VCModel(Y, X, V)
Σ̂path, β̂path, = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=30)
Σ = fill(0.5, length(V))
resetModel!(vcm, Σ)
```
"""
function resetModel!(
    vcm :: VCModel,
    Σ :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}} 
    ) where {T <: Real}

    vcm.Σ .= Σ
    updateΩ!(vcm)
    updateΩest!(vcm)
    update_arrays!(vcm)
    vcm.R .= reshape(vcm.ΩinvY, size(vcm))
end 

"""
    resetModel!(vcm::VCintModel, [Σ, Σint])

Reset [`VCintModel`](@ref) with initial estimates `Σ` and `Σint` (if given). If unspecified, 
it is set to a vector of ones.
"""
function resetModel!(
    vcm  :: VCintModel,
    Σ    :: AbstractVector = ones(ngroups(vcm) + 1),
    Σint :: AbstractVector = ones(ngroups(vcm)) 
    ) 
   
    vcm.Σ .= Σ
    vcm.Σint .= Σint 
    updateΩ!(vcm)
    updateΩest!(vcm)
    update_arrays!(vcm)

end 

include("vcselect.jl")
include("vcselect_interact.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalgops.jl")

end # module