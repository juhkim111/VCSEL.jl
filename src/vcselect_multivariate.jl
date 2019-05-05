export vcselect, vcselectpath

"""
    vcselect(Y, V, penfun, λ, penwt, Σ)
Minimize penalized log-likelihood of variance component model with multivariate
response. The objective function is `0.5logdet(Ω) + 0.5(vecY)'*inv(Ω)*(vecY) +
λ * sum(penwt.*penfun(√tr(Σ)))`, where `Ω = Σ[1]⊗V[1] + ... + Σ[end]⊗V[end]`.
# Input
- `Y::Vector`: response matrix
- `V::Vector{Matrix}`: covariance matrices
# Keyword 
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty()
- `λ`: penalty strength
- `penwt::Vector`: vector of penalty weights. penwt[end] must equal to 0.
- `Σ::Vector`: initial values
- `maxiter::Int`: maximum number of iterations
- `tolfun`: tolerance in objective value
- `verbose`: display switch
# Output
- `Σ`: minimizer
- `obj`: objevtive value at the minimizer
- `niter`: number of iterations
- `Ω`: covariance matrix evaluated at the minimizer
- `Ωinv`: precision (inverse covariance) matrix evaluated at the minimizer
"""
function vcselect(
    Y       :: AbstractMatrix{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = one(T),
    penwt   :: AbstractVector = [ones(T, length(V)-1); zero(T)],
    Σ       :: AbstractArray = fill(Matrix(1.0I, size(Y, 2), size(Y, 2)), length(V)),
    Ω       :: AbstractMatrix{T} = zeros(T, size(V[1])), 
    Ωinv    :: AbstractMatrix{T} = zeros(T, size(V[1])),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}

    # initialize algorithm 
    n, d = size(Y)
    nvarcomps = length(V) 
    fill!(Ω, 0)     # covariance matrix 
    for j in 1:length(V)
        Ω .+= σ2[j] .* V[j]
    end


    # reshape 
    R = reshape(Ωinv * vec(Y), n, d)

    # `Ikron1 = I_d ⊗ 1_n` 
    Ikron1 = kron(Matrix(I, d, d), ones(n))

    # MM loop 
    for i in 1:nvarcomps
        # `tmp = (Ikron1)' * [kron(ones(d, d), vcobs.V[i]) .* Ωinv] * (Ikron1)`
        tmp = kron(ones(d, d), vcobs.V[i]) .* Ωinv
        lmul!(Ikron1', tmp), rmul!(tmp, Ikron1)

        # add penalty unless it's the last variance component 
        if isa(penfun, L1Penalty) && i < nvarcomps
            # `tmp += (penstrength / tr(vcm.Σ[i])) * Matrix(I, d, d)`
            axpy!((λ * penwt[i] / tr(vcm.Σ[i])), Matrix(I, d, d), tmp)
            # or use for loop to add diagonal elemetns 
            # tmpconst = λ * penwt[i] / tr(vcm.Σ[i])
            # for j in 1:d
            #     tmp[j, j] .+= tmpconst  
            # end             
        end 

        L = cholesky!(Hermitian(tmp)).L
        Linv = inv(L)

        # `vcm.Σ[i] = Linv' * sqrt((R*vcm.Σ[i]*L)' * vcobs.V[i] * (R*vcm.Σ[i]*L)) * Linv`
        tmp2 = R * (vcm.Σ[i] * L)
        vcm.Σ[i] = sqrt(BLAS.gemm('T', 'N', tmp2, vcobs.V[i] * tmp2))
        lmul!(Linv', vcm.Σ[i]), rmul!(vcm.Σ[i], Linv)
    end 
    


    

end 
"""
function mm_update_Σ!(
    vcm    :: T1,
    vcobs  :: T2,
    Ωinv   :: AbstractMatrix,
    penfun :: Penalty,
    λ      :: Real,
    penwt  :: AbstractVector
    ) where {
        T1 <: VarianceComponentModel,
        T2 <: VarianceComponentVariate
    }

    # size of Y 
    n, d = size(vcobs)

    # reshape 
    R = reshape(Ωinv * vec(vcobs.Y), n, d)

    # `Ikron1 = I_d ⊗ 1_n` 
    Ikron1 = kron(Matrix(I, d, d), ones(n))

    # MM loop 
    for i in 1:nvarcomps(vcm)
        # `tmp = (Ikron1)' * [kron(ones(d, d), vcobs.V[i]) .* Ωinv] * (Ikron1)`
        tmp = kron(ones(d, d), vcobs.V[i]) .* Ωinv
        lmul!(Ikron1', tmp), rmul!(tmp, Ikron1)

        # add penalty unless it's the last variance component 
        if i < nvarcomps(vcm)
            # `tmp += (penstrength / tr(vcm.Σ[i])) * Matrix(I, d, d)`
            axpy!((λ * penwt[i] / tr(vcm.Σ[i])), Matrix(I, d, d), tmp)
        end 
        L = cholesky(Hermitian(tmp)).L
        Linv = inv(tmp)

        # `vcm.Σ[i] = Linv' * sqrt((R*vcm.Σ[i]*L)' * vcobs.V[i] * (R*vcm.Σ[i]*L)) * Linv`
        tmp2 = R * (vcm.Σ[i] * L)
        vcm.Σ[i] = sqrt(BLAS.gemm('T', 'N', tmp2, vcobs.V[i] * tmp2))
        lmul!(Linv', vcm.Σ[i]), rmul!(vcm.Σ[i], Linv)
    end 
    
end 

# """
#     mm_update_Σ!(vcm, vcobs, penfun, λ, penwt)

# Update Σ using MM algorithm, for multivariate response model  
# """
# function mm_update_Σ!(
#     vcm    :: T1,
#     vcobs  :: T2,
#     penfun :: Penalty,
#     λ      :: Real,
#     penwt  :: AbstractVector
#     ) where {
#         T1 <: VarianceComponentModel,
#         T2 <: VarianceComponentVariate
#     }

#     # size of Y 
#     n, d = size(vcobs)

#     # construct covariance matrix 
#     Ω = cov(vcm, vcobs)

#     # obtain inverse of Ω using cholesky factorization 
#     Ωchol = cholesky(Hermitian(Ω))
#     Ωinv = inv(Ωchol) # inverse of Ω

#     # 
#     mm_update_Σ!(vcm, vcobs, Ωinv, penfun, λ, penwt)
# end 