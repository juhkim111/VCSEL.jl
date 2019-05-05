export vcselect, 

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
    Y             :: AbstractMatrix{T},
    V             :: AbstractVector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector = [ones(T, length(V)-1); zero(T)],
    Σ             :: AbstractArray = fill(Matrix(1.0I, size(Y, 2), size(Y, 2)), length(V)),
    Ω             :: AbstractMatrix{T} = zeros(T, prod(size(Y)), prod(size(Y))), 
    Ωinv          :: AbstractMatrix{T} = zeros(T, Ω),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-6,
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true 
    ) where {T <: Real}

    # check frob norm equals to 1 
    if checkfrobnorm 
        checkfrobnorm!(V)
    end 

    # initialize algorithm 
    n, d = size(Y)          # size of response matrix 
    nvarcomps = length(V)   # no. of variance components
    fill!(Ω, 0)             # covariance matrix 
    for j in 1:nvarcomps
        kronaxpy!(Σ[j], V[j], Ω)
    end
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv[:] = inv(Ωchol) 
    vecY = vec(Y)
    v = Ωinv * vecY   
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(vecY, v) # objective value 
    pen = 0.0
    for j in 1:(nvarcomps - 1)
        pen += penwt[j] * value(penfun, √tr(Σ[j]))
    end
    loglConst = (1//2) * n * d * log(2π)
    obj += loglConst + λ * pen

    # reshape 
    R = reshape(Ωinv * vecY, n, d)

    # `I_d ⊗ 1_n` 
    kron_I_one = kron(Matrix(I, d, d), ones(n))

    # `1_d 1_d' ⊗ V[i]`
    kron_ones_V = similar(V)
    for i in 1:nvarcomps 
        kron_ones_V[i] = kron(ones(d, d), V[i])
    end 

    # pre-allocate memory 
    W = similar(Ω)
    Linv = similar(Σ[1])

    # display 
    if verbose
        println("iter = 0")
        #println("Σ   = ", Σ)
        println("obj  = ", obj)
        objvec = obj
    end  

    # MM loop 
    niters = 0 
    for iter in 1:maxiter 
        fill!(Ω, 0)
        # update variance components
        for i in 1:nvarcomps
            # `W = (kron_I_one)' * [kron(ones(d, d), V[i]) .* Ωinv] * (kron_I_one)`
            W = kron_ones_V[i] .* Ωinv
            lmul!(kron_I_one', W), rmul!(W, kron_I_one)

            # add penalty unless it's the last variance component 
            if isa(penfun, L1Penalty) && i < nvarcomps
                # `tmp += (penstrength / tr(vcm.Σ[i])) * Matrix(I, d, d)`
                # axpy!((λ * penwt[i] / tr(Σ[i])), Matrix(I, d, d), tmp)
                # or use for loop to add diagonal elemetns 
                penconst = λ * penwt[i] / tr(Σ[i])
                for j in 1:d
                    W[j, j] .+= penconst  
                end             
            end 

            L = cholesky!(Symmetric(W))
            Linv[:] = inv(L)

            # `vcm.Σ[i] = Linv' * sqrt((R*vcm.Σ[i]*L)' * vcobs.V[i] * (R*vcm.Σ[i]*L)) * Linv`
            W = R * (Σ[i] * L.L)
            Σ[i] = sqrt(BLAS.gemm('T', 'N', W, V[i] * W))
            lmul!(Linv', Σ[i]), rmul!(Σ[i], Linv)
        end 

        # update Ω
        for j in 1:nvarcomps 
            kronaxpy!(Σ[j], V[j], Ω)
        end

        # update Ωchol, Ωinv, v, R
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(v, Ωinv, vecY)
        R = reshape(Ωinv * vecY, n, d)

        # update objective value 
        objold = obj 
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(vecY, v) # objective value 
        pen = 0.0
        for j in 1:(nvarcomps - 1)
            pen += penwt[j] * value(penfun, √tr(Σ[j]))
        end
        obj += loglConst + λ * pen

        # display current iterate if specified 
        if verbose
            println("iter = ", iter)
            println("obj  = ", obj)
            objvec = [objvec; obj] 
        end

        # check convergence
        if abs(obj - objold) < tol * (abs(objold) + 1)
            niters = iter
            break
        end
    end # end of iteration 
    
    # construct final Ω matrix
    fill!(Ω, 0)
    for j in 1:nvarcomps 
        kronaxpy!(Σ[j], V[j], Ω)
    end
    
    # output
    if niters == 0
        niters = maxiter
    end

    if verbose 
        return Σ, obj, niters, Ω, objvec;
    else 
        return Σ, obj, niters, Ω;
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