export vcselect

"""
    vcselect(Y, V; penfun, λ, penwt, Σ)

Minimize penalized log-likelihood of variance component model with multivariate
response. The objective function to minimize is
    `0.5n*d*log(2π) + 0.5logdet(Ω) + 0.5(vecY)'*inv(Ω)*(vecY) + λ * sum(penwt.*penfun(√tr(Σ)))`, 
where `Ω = Σ[1]⊗V[1] + ... + Σ[end]⊗V[end]`, `V[end] = I`, and `(n,d)` is dimension of `Y`.
Minimization is achieved via majorization-minimization (MM) algorithm. 

# Input
- `Y`: response matrix
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default 

# Keyword 
- `penfun`: penalty function, e.g., `NoPenalty()` (default), `L1Penalty()`, `MCPPenalty(γ = 2.0)`
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights where penwt[end] must equal to 0, 
        default is (1,1,...,1,0)
- `Σ`: initial values, default is (I,I,...,I) where I is dxd identity matrix
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 
- `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
        default is true 

# Output
- `Σ`: vector of estimated variance components
- `obj`: objective value at the estimated variance components
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `objvec`: vector of objective values at each iteration, 
    returned only if `verbose` is true
"""
function vcselect(
    Y             :: AbstractMatrix{T},
    V             :: AbstractVector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector = [ones(T, length(V)-1); zero(T)],
    Σ             :: AbstractArray = fill(Matrix(one(T) * I, size(Y, 2), size(Y, 2)), length(V)),
    Ω             :: AbstractMatrix{T} = zeros(T, prod(size(Y)), prod(size(Y))), 
    Ωinv          :: AbstractMatrix{T} = zeros(T, prod(size(Y)), prod(size(Y))),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-6,
    atol          :: AbstractFloat = 1e-16,
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true 
    ) where {T <: Number}

    # check frob norm equals to 1 
    if checkfrobnorm 
        checkfrobnorm!(V)
    end 

    # 
    zeroT = zero(T)
    ϵ = convert(T, 1e-8)

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
    ΩinvY = Ωinv * vecY   
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(vecY, ΩinvY) # objective value 
    pen = 0.0
    for j in 1:(nvarcomps - 1)
        pen += penwt[j] * value(penfun, √tr(Σ[j]))
    end
    loglConst = (1//2) * n * d * log(2π)
    obj += loglConst + λ * pen

    # reshape 
    R = reshape(ΩinvY, n, d)

    # `I_d ⊗ 1_n` 
    kron_I_one = kron(Matrix(I, d, d), ones(n))

    # `1_d 1_d' ⊗ V[i]`
    kron_ones_V = similar(V)
    for i in 1:nvarcomps 
        kron_ones_V[i] = kron(ones(d, d), V[i])
    end 

    # pre-allocate memory 
    W = similar(Ω)
    Linv = zeros(d, d)
    tmp = similar(Y)

    # display 
    if verbose
        println("iter = 0")
        println("obj  = ", obj)
        objvec = obj
    end  

    # MM loop 
    niters = 0 
    for iter in 1:maxiter 
        # update variance components
        for i in 1:nvarcomps

            # if previous iterate Σ[i] is close to zero matrix, move on to the next Σ[i+1]
            if isapprox(Σ[i], zeros(d, d)) #; atol=atol)
                continue 
            end 

            # `W = (kron_I_one)' * [kron(ones(d, d), V[i]) .* Ωinv] * (kron_I_one)`
            W = kron_ones_V[i] .* Ωinv
            W = BLAS.gemm('T', 'N', kron_I_one, W * kron_I_one)

            # add penalty unless it's the last variance component 
            if isa(penfun, L1Penalty) && i < nvarcomps
                penconst = λ * penwt[i] / √tr(Σ[i])
                for j in 1:d
                    W[j, j] += penconst  
                end             
            end 

            L = cholesky!(Symmetric(W)).L
            Linv[:] = inv(L)

            # `vcm.Σ[i] = Linv' * sqrt((R*vcm.Σ[i]*L)' * vcobs.V[i] * (R*vcm.Σ[i]*L)) * Linv`
            tmp = R * (Σ[i] * L)
            tmp = BLAS.gemm('T', 'N', tmp, V[i] * tmp)
            storage = eigen!(Symmetric(tmp))
            # if negative eigenvalue, set it to 0 
            @inbounds for k in 1:d
                storage.values[k] = storage.values[k] > zeroT ? √storage.values[k] : zeroT
            end 
            Σ[i] = BLAS.gemm('N', 'T', storage.vectors * Diagonal(storage.values), storage.vectors)
            Σ[i] = BLAS.gemm('T', 'N', Linv, Σ[i] * Linv)

        end 

        # make sure the last variance component is pos. def.
        clamp_diagonal!(Σ[end], ϵ, T(Inf))

        # update variance component unless zero matrix 
        fill!(Ω, 0)
        for j in 1:nvarcomps
            if isapprox(Σ[j], zeros(d, d); atol=atol)
                continue 
            end 
            kronaxpy!(Σ[j], V[j], Ω)
        end 

        # update Ωchol, Ωinv, v, R
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(ΩinvY, Ωinv, vecY)
        R = reshape(ΩinvY, n, d)

        # update objective value 
        objold = obj 
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(vecY, ΩinvY) # objective value 
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
        if Σ[j] == zeros(d, d)
            continue 
        end 
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

