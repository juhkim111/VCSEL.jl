__precompile__()

module VCSEL

using PenaltyFunctions, LinearAlgebra, StatsBase, Distributions, Reexport, Plots
using IterativeSolvers
@reexport using PenaltyFunctions
@reexport using Plots 

import Base: size, length, +
export 
# struct 
    VCModel, MvcCovMatrix,
# operations 
    clamp_diagonal!,
    kronaxpy!,
    nullprojection,  
    objvalue,  
# maximum lambda 
    maxlambda, 
# utilities function 
    ncovariates, nvarcomps, nmeanparams, ngroups,
    plotsolpath, resetModel!, 
    rankvarcomps,   
    update_arrays!,
    updateM!,
# algorithm 
    mm_update_σ2!, mm_update_Σ!, mm_update_Σ_v2!,
    vcselect, vcselect!, vcselectpath, vcselectpath!

"""
    MvcCovMatrix(Σ, G)
"""
struct MvcCovMatrix{T <: Real} <: AbstractMatrix{T}
    Σ  :: Vector{Matrix{T}} # m + 1 variance components: Σ1, ..., Σm, Σ0
    G  :: Vector{Matrix{T}} # m genotype matrices: G1, ..., Gm
    # working arrays
    storage_nd   :: Matrix{T}
    storage_qd_1 :: Matrix{T}
    storage_qd_2 :: Matrix{T}
end

"""
    MvcCovMatrix(Σ, G)

Construct [`MvcCovMatrix`](@ref) from  
"""
function MvcCovMatrix(Σ::Vector{Matrix{T}}, G::Vector{Matrix{T}}) where T <: Real
    n, d, m      = size(G[1], 1), size(Σ[1], 1), length(G)
    storage_nd   = Matrix{T}(undef, n, d)
    storage_qd_1 = Matrix{T}(undef, maximum([size(G[i], 2) for i in 1:m]), d)
    storage_qd_2 = similar(storage_qd_1)
    MvcCovMatrix{T}(Σ, G, storage_nd, storage_qd_1, storage_qd_2)
end



"""
    VCModel
"""
struct VCModel{T <: Real} 
    # data
    Yobs        :: AbstractVecOrMat{T}
    Xobs        :: AbstractVecOrMat{T}
    Gobs        :: AbstractVector{Matrix{T}}
    Y           :: AbstractVecOrMat{T}       # projected response matrix 
    vecY        :: AbstractVector{T}         # vectorized projected response matrix 
    G           :: AbstractVector{Matrix{T}} # projected V
    # model parameters 
    β           :: AbstractVecOrMat{T}
    Ω           :: MvcCovMatrix{T}
    # covariance matrix and working arrays 
    vecR        :: AbstractVector{T}
    R           :: AbstractVecOrMat{T}
    storage_nd  :: AbstractVector{T}
    storage_nd_q :: AbstractVecOrMat{T}
    M           :: AbstractVecOrMat{T}
    L           :: AbstractVecOrMat{T}
    Linv        :: AbstractVecOrMat{T}
    storage_q_d :: AbstractVector{Matrix{T}}
    Mdxd        :: AbstractVecOrMat{T}
    Σtmp        :: AbstractVector{Matrix{T}}
end


"""
    VCModel(Yobs, Xobs, Gobs, [Σ])

Default constructor of [`VCModel`](@ref) type.
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Xobs  :: AbstractVecOrMat{T},
    Gobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = [Matrix(one(T)*I, 
            size(Yobs, 2), size(Yobs, 2)) for i in 1:(length(Gobs) + 1)]
    ) where {T <: AbstractFloat}

    # handle error 
    @assert length(Gobs) == (length(Σ) - 1) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # projection 
    Y, G, = nullprojection(Yobs, Xobs, Gobs)
    n, d = size(Y)
    m = length(Gobs)
    vecY = vec(Y)
    p = size(Xobs, 2)
    # 
    β = Matrix{T}(undef, p, d)
    #
    Ω = MvcCovMatrix(Σ, G)
    # pre-allocate working arrays 
    vecR = cg(Ω, vecY)
    R = reshape(vecR, n, d)
    # 
    storage_nd = Vector{T}(undef, n * d)
    storage_nd_q = ones(T, n*d, maximum([size(G[i], 2) for i in 1:m]))
    M = Matrix{T}(undef, d, d)
    L = Matrix{T}(undef, d, d)
    Linv = Matrix{T}(undef, d, d)
    storage_q_d = [Matrix{T}(undef, 
            size(G[i], 2), d) for i in 1:m]
    Mdxd = Matrix{T}(undef, d, d) # d x d 
    # 
    Σtmp = [Matrix{T}(undef, d, d) for i in 1:(length(Gobs) + 1)]

    # 
    VCModel{T}(Yobs, Xobs, Gobs, Y, vecY, G, β, Ω, vecR, R, 
                storage_nd, storage_nd_q, M, L, Linv, storage_q_d, Mdxd, Σtmp)

end 

"""
    VCModel(obs, Gobs, [Σ])

Construct [`VCModel`](@ref) from `Y` and `G`. `X` is treated empty. 
"""
function VCModel(
    Yobs  :: AbstractMatrix{T},
    Gobs  :: AbstractVector{Matrix{T}},
    Σ     :: AbstractVector{Matrix{T}} = [Matrix(one(T)*I, 
            size(Yobs, 2), size(Yobs, 2)) for i in 1:(length(Gobs) + 1)]
    ) where {T <: Real}

    # handle error 
    @assert length(Gobs) == (length(Σ)-1) "vector of covariance matrices and vector of variance components should have the same length!\n"

    # create empty matrix for Xobs 
    Xobs = zeros(T, size(Yobs, 1), 0) 

    # call VCModel constructor 
    VCModel(Yobs, Xobs, Gobs, Σ)
end 


"""
    length(vcm::VCModel)

Length `d` of response. 
"""
length(vcm::VCModel) = size(vcm.Y, 2)


"""
    ncovariates(vcm::VCModel)
    ncovariates(vcm::VCintModel)

Number of fixed effects parameters `p` of a [`VCModel`](@ref) or [`VCintModel`](@ref). 
"""
ncovariates(vcm::VCModel) = size(vcm.Xobs, 2)

"""
    size(vcm::VCModel)
    size(vcm::VCintModel)

Size `(n, d)` of response matrix of a [`VCModel`](@ref) or [`VCintModel`](@ref). 
"""
size(vcm::VCModel) = size(vcm.Yobs)

"""
    size(M::MvcCovMatrix)

Size of [`MvcCovMatrix`](@ref)
"""
LinearAlgebra.issymmetric(::MvcCovMatrix) = true
size(M::MvcCovMatrix) = 
    (size(M.Σ[1], 1) * size(M.G[1], 1), size(M.Σ[1], 1) * size(M.G[1], 1))


"""
    nmeanparams(vcm::VCModel)
    nmeanparams(vcm::VCintModel)

Number of mean parameters `p * d` of [`VCModel`](@ref) or [`VCintModel`](@ref).
"""
nmeanparams(vcm::VCModel) = length(vcm.β)

"""
    nvarcomps(vcm::VCModel)
    nvarcomps(vcm::VCintModel)

Number of variance components. 
""" 
nvarcomps(vcm::VCModel) = length((vcm.Ω).Σ) 

""" 
    ngroups(vcm::VCModel)

Number of groups, `m`, for [`VCModel`](@ref) or [`VCintModel`](@ref).
"""
ngroups(vcm::VCModel) = length(vcm.G)



# """
#     updateΩest!(vcm::VCModel)

# Update covariance matrix `Ωest` for [`VCModel`](@ref). `Ωest` has the same dimension as `Vobs`. 
# """
# function updateΩest!(vcm::VCModel)

#     if isempty(vcm.Xobs) 
#         vcm.Ωest .= vcm.Ω
#     else
#         fill!(vcm.Ωest, 0)
#         for k in 1:nvarcomps(vcm)
#             kronaxpy!(vcm.Σ[k], vcm.Vobs[k], vcm.Ωest)
#         end
#     end 
#     vcm.Ωest
# end



"""
    update_arrays!(vcm, Σnew)

Update working arrays `Ωchol`, `Ωinv`, `ΩinvY`, `R`.
"""
function update_arrays!(vcm::Union{VCModel}, Σnew::Matrix{T}) where T <: Real

    (vcm.Ω).Σ .= Σnew
    cg!(vcm.vecR, vcm.Ω, vcm.vecY)
    vcm.R = reshape(vcm.vecR, n, d)
    nothing 
end 

# """
#     resetModel!(vcm::VCModel)

# Reset [`VCModel`](@ref) with initial estimates `Σ`. If `Σ` is unspecified, 
# it is set to a vector of ones or identity matrices based on its dimension.

# # Example

# ```julia
# vcm = VCModel(Y, X, V)
# Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=50)
# resetModel!(vcm)
# ```
# """
# function resetModel!(
#     vcm :: VCModel
#     ) 
#     d = length(vcm)
#     if typeof(vcm.Σ[1]) <: Matrix 
#         resetModel!(vcm, 
#                 [Matrix(one(eltype(vcm.Σ[1]))*I, d, d) for i in eachindex(vcm.Σ)])
#     else 
#         resetModel!(vcm,
#                 ones(eltype(vcm.Σ[1]), nvarcomps(vcm)))
#     end 
# end 

# """
#     resetModel!(vcm::VCModel, Σ)

# Reset [`VCModel`](@ref) with initial estimates `Σ`.

# # Example

# ```julia
# vcm = VCModel(Y, X, V)
# Σ̂path, β̂path, = vcselectpath!(vcm; penfun=MCPPenalty(), nλ=30)
# Σ = fill(0.5, length(V))
# resetModel!(vcm, Σ)
# ```
# """
# function resetModel!(
#     vcm :: VCModel,
#     Σ :: Union{AbstractVector{T}, AbstractVector{Matrix{T}}} 
#     ) where {T <: Real}


#     # pre-allocate working arrays 
#     vecR = Vector{T}(undef, n * d)
#     cg!(vecR, Ω, vecY)
#     R = reshape(vecR, n, d)

#     vcm.R .= reshape(vcm.ΩinvY, size(vcm))
# end 



include("vcselect.jl")
include("vcselect_interact.jl")
include("maxlambda.jl")
include("utilities.jl")
include("linalgops.jl")

end # module