module VarianceComponentSelect

using PenaltyFunctions
using LinearAlgebra 
using StatsBase
using Distributions 

export vcselect, vcmm, maxlambda

include("vcmm.jl")
include("maxlambda.jl")


"""
    vcselect(yobs, Xobs, Vobs; penfun=NoPenalty(), penwt=[ones(length(Vobs)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(Vobs)), maxiter=1000, tolfun=1e-6)

Generate solution path of variance components along varying lambda values.

# Input  
- `yobs::Vector{Float64}`: response vector. 
- `Xobs::Matrix{Float64}`: covariate matrix.
- `Vobs::Vector{Matrix{Float64}}`: vector of covariance matrices; (V1,...,Vm).
- `penfun::Penalty`: penalty function. Default is NoPenalty().
- `penwt::Vector{Float64}`: weights for penalty term. Default is (1,1,...1,0).
- `nlambda::Int`: number of tuning parameter values. Default is 100. 
- `λpath::Vector{Float64}`: user-supplied grid of tuning parameter values. 
        If unspeficied, internally generate a grid.
- `σ2::Vector{Float64}`: initial estimates.
- `maxiter::Int`: maximum number of iteration for MM loop.
- `tolfun::Float64`: tolerance in objective value for MM loop.
- `verbose::Bool`: display switch. 

# Output 
- `σ2path`: solution path along varying lambda values. 
        Each column gives estimated σ2 at specific lambda.
- `objpath`: objective value path. 
- `λpath`: tuning parameter path.
"""
function vcselect(
    yobs    :: Vector{Float64},
    Xobs    :: Matrix{Float64},
    Vobs    :: Vector{Matrix{Float64}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: Vector{Float64} = [ones(length(Vobs)-1); 0.0],
    nlambda :: Int = 100, 
    λpath   :: Vector{Float64} = Float64[],
    σ2      :: Vector{Float64} = ones(length(Vobs)),
    maxiter :: Int = 1000,
    tol     :: Float64 = 1e-6,
    verbose :: Bool = false
    ) 

    # number of groups 
    m = length(Vobs) - 1

    ## REML: find B s.t. columns of B span the null space of X' and B'B = I
    # pivoted QR factorization of I-X(X'X)^{-1}X'
    QRfact = qr(I - Xobs * inv(cholesky(Xobs' * Xobs)) * Xobs', Val(true))
    # extract orthonormal basis of C(I-P)
    B = QRfact.Q[:, abs.(diag(QRfact.R)) .> tol] 
    # REML transformed response vector 
    y = B' * yobs 
    # REML transformed covariance matrices 
    V  = Array{Matrix{Float64}}(undef, m + 1)
    for i in 1:(m + 1)
        V[i] = B' * Vobs[i] * B  
    end  

    # make sure frobenius norm of Vi equals to 1 
    for i in 1:(m + 1)
        if norm(V[i]) != 1
            V[i] ./= norm(V[i])
        end 
    end 

    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ = maxlambda(y, V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        end 

        # initialize solution path 
        σ2path = zeros(m + 1, nlambda)
        objpath = zeros(nlambda)

        # create solution path 
        for iter in 1:length(λpath)
            λ = λpath[iter]
            σ2path[:, iter], objpath[iter], = vcmm(y, V; penfun=penfun, λ=λ, penwt=penwt,  
                        σ2=σ2, maxiter=maxiter, tolfun=tol, verbose=verbose)
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, = vcmm(y, V; penfun=penfun)
    end 

    # output 
    return σ2path, objpath, λpath 


end 


end # module
