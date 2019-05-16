export vcselect, vcselectpath 

"""
    vcselect(Y, X, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Project covariate matrix `X` to null space of `X'` and 
call `vcselect(Y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)`

# Input
- `Y`: response matrix 
- `X`: covariate matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
    note (1) V[end] should be identity matrix or identity matrix divided by √n
    note (2) each V[i] needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each V[i] by its frobenius norm  

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
- `β`: estimated fixed effects parameter, using REML estimates
- `obj`: objective value at the estimated variance components
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `objvec`: vector of objective values at each iteration, 
    returned only if `verbose` is true
"""
function vcselect( 
    Y             :: AbstractMatrix{T},
    X             :: AbstractMatrix{T},
    V             :: AbstractVector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector = [ones(T, length(V)-1); zero(T)],
    Σ             :: AbstractArray = fill(Matrix(one(T) * I, size(Y, 2), size(Y, 2)), length(V)),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-6,
    atol          :: AbstractFloat = 1e-16,
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true 
    ) where {T <: Real}

    if checkfrobnorm 
        # check frob norm equals to 1 
        checkfrobnorm!(V)
    end 

    # project onto null space 
    Ynew, Vnew = nullprojection(Y, X, V)

    # call vcselect 
    if verbose 
        Σ, obj, niters, _, objvec = vcselect(Ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                            Σ=Σ, maxiter=maxiter, tol=tol, atol=atol, verbose=verbose,
                            checkfrobnorm=false)
    else 
        Σ, obj, niters, = vcselect(Ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                            Σ=Σ, maxiter=maxiter, tol=tol, atol=atol, verbose=verbose,
                            checkfrobnorm=false)
    end 

    # update Ω with estimated variance components 
    n, d = size(Y)
    Ω = zeros(T, n*d, n*d)
    for j in eachindex(Σ)
        if Σ[j] == zeros(d, d)
            continue 
        end 
        kronaxpy!(Σ[j], V[j], Ω)
    end

    # estimate fixed effects 
    β = fixedeffects(Y, X, Ω)

    # return output 
    if verbose 
        return Σ, β, obj, niters, Ω, objvec;
    else 
        return Σ, β, obj, niters, Ω;
    end 

end 

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
    ) where {T <: Real}

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

"""
    vcselectpath(y, X, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Project `Y` to null space of `X` and generate solution path of variance components 
along varying lambda values.

# Input  
- `Y`: response matrix
- `X`: covariate matrix 
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nlambda`: number of tuning parameter values, default is 100
- `λpath`: user-supplied grid of tuning parameter values
    If unspeficied, internally generate a grid
- `Σ`: initial estimates
- `maxiter`: maximum number of iteration for MM loop
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 
- `fixedeffects`: whether user wants fixed effects parameter 
    to be estimated and returned, default is false 

# Output 
- `Σpath`: matrix of estimated variance components at each tuning parameter `λ`,
    each column gives vector of estimated variance components `Σ` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
- `niterspath`: vector of no. of iterations to convergence 
- `βpath`: matrix of estimated fixed effects at each tuning parameter `λ`
"""
function vcselectpath(
    Y            :: AbstractMatrix{T},
    X            :: AbstractMatrix{T},
    V            :: AbstractVector{Matrix{T}};
    penfun       :: Penalty = NoPenalty(),
    penwt        :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda      :: Int = 100, 
    λpath        :: AbstractVector{T} = T[],
    Σ            :: AbstractArray = fill(Matrix(one(T) * I, size(Y, 2), size(Y, 2)), length(V)),
    maxiter      :: Int = 1000,
    tol          :: AbstractFloat = 1e-6,
    atol         :: AbstractFloat = 1e-16,
    verbose      :: Bool = false,
    fixedeffects :: Bool = false 
    ) where {T <: Real}

    # project y and V onto nullspace of X
    ynew, Vnew = nullprojection(Y, X, V)

    # 
    Σpath, objpath, λpath, niterspath = vcselectpath(ynew, Vnew;
        penfun=penfun, penwt=penwt, nlambda=nlambda, λpath=λpath, σ2=σ2, maxiter=maxiter,
        tol=tol, verbose=verbose)

    # if user wants fixed effects estimates, estimate β
    if fixedeffects 
        βpath = zeros(T, size(X, 2), nlambda)
        for iter in 1:length(λpath)
            βpath[:, iter] = fixedeffects(y, X, V, view(Σpath, :, iter))
        end 

        # output 
        return Σpath, objpath, λpath, niterspath, βpath
    else 
        return Σpath, objpath, λpath, niterspath
    end 
end 


"""
    vcselectpath(Y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Generate solution path of variance components along varying lambda values.

# Input
- `Y`: response matrix
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nlambda`: number of tuning parameter values, default is 100
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid
- `Σ`: initial estimates.
- `maxiter`: maximum number of iteration for MM loop.
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output 
- `Σpath`: matrix of estimated variance components at each tuning parameter `λ`,
each column gives vector of estimated variance components `Σ` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
- `niterspath`: vector of no. of iterations to convergence
"""
function vcselectpath(
    Y       :: AbstractMatrix{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = T[],
    Σ       :: AbstractArray = fill(Matrix(one(T) * I, size(Y, 2), size(Y, 2)), length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    atol    :: AbstractFloat = 1e-16,
    verbose :: Bool = false
    ) where {T <: Real}

    checkfrobnorm!(V)

    ## generate solution path based on penalty 
    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ = maxlambda(Y, V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        else # if lambda grid specified, make sure nlambda matches 
            nlambda = length(λpath)
        end 

        # initialize arrays  
        Σpath = zeros(T, length(V), nlambda)
        objpath = zeros(T, nlambda)
        niterspath = zeros(Int, nlambda)

        # create solution path 
        for iter in 1:nlambda 
            Σ, objpath[iter], niterspath[iter], = 
                    vcselect(Y, V; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                    σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose, checkfrobnorm=false)
            Σpath[:, iter] = Σ
        end

    else # if no penalty, there is no lambda grid 
        Σpath, objpath, niterspath, = vcselect(Y, V; 
                penfun=penfun, maxiter=maxiter, tol=tol, verbose=verbose, 
                checkfrobnorm=false)
    end 

    # output 
    return Σpath, objpath, λpath, niterspath


end 