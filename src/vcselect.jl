"""
    vcselect(y, X, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Project covariate matrix `X` to null space of `X'` and 
call `vcselect(y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)`

# Input
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
        note that V[end] should be identity matrix

# Keyword
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty()
        γ for MCPPenalty() is 2.0 by default,
        if different γ (say 3.0) is desired, simply do MCPPenalty(3.0)
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `σ2`: initial values, default is (1,1,...,1)
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output
- `σ2`: vector of estimated variance components 
- `beta`: estimated fixed effects parameter, using REML estimates
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `Ωinv`: precision (inverse covariance) matrix evaluated at the estimated variance components
"""
function vcselect( 
    y       :: AbstractVector{T},
    X       :: AbstractMatrix{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = one(T),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}

    ynew, Vnew = nullprojection(y, X, V)

    # call vcselect 
    σ2, obj, niters = vcselect(ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                            σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)

    # update Ω with estimated variance components 
    Ω = zeros(T, size(V[1]))
    for i in eachindex(σ2)
        if iszero(σ2[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    # inverse using cholesky 
    Ωchol = cholesky(Symmetric(Ω))

    # estimate fixed effects: pinv(X'*Ωinv*X)(X'*Ωinv*y)
    XtΩinvX = BLAS.gemm('T', 'N', X, Ωchol \ X)
    beta = BLAS.gemv('T', X, Ωchol \ y) # overwriting Ωinv with X'*Ωinv

    beta = pinv(XtΩinvX) * beta

    return σ2, beta, obj, niters, Ω;
end

"""
    vcselect(y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]` and `V[end] = I`
Minimization is achieved via majorization-minimization (MM) algorithm. 

# Input
- `y`: response vector
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
        note that V[end] should be identity matrix

# Keyword
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty(γ = 2.0)
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `σ2`: initial values, default is (1,1,...,1)
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output
- `σ2`: vector of estimated variance components 
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `Ωinv`: precision (inverse covariance) matrix evaluated at the estimated variance components
"""
function vcselect( 
    y       :: AbstractVector{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = one(T),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real} 

    # exception handling 
    #ArgumentError() # values in penwt should be nonnegative 
    #ArgumentError() # previous σ2 values should be nonnegative 


    # initialize algorithm
    n = length(y)       # no. observations
    m = length(V) - 1   # no. variance components
    Ω = zeros(n, n)     # covariance matrix 
    for j in 1:(m + 1)
        Ω .+= σ2[j] .* V[j]
    end
  
    Ωchol = cholesky(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    v = Ωinv * y
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v) # objective value 
    pen = 0.0
    for j in 1:m
        pen += penwt[j] * value(penfun, √σ2[j])
    end
    loglConst = (1//2) * n * log(2π) # (has to be n-r; refer to Searle, Casella)
    obj += loglConst + λ * pen
    if verbose
        println("iter = 0")
        println("σ2   = ", σ2)
        println("obj  = ", obj)
    end

    # pre-allocate memory 
    w = similar(v)

    # MM loop 
    niters = 0
    for iter in 1:maxiter
        # update variance components
        fill!(Ω, 0)
        for j in 1:m
            # move onto the next variance component if previous iterate is 0
            if iszero(σ2[j]) 
                continue 
            # set to 0 and move onto the next variance component if penalty weight is 0
            elseif iszero(penwt[j]) 
                σ2[j] = zero(T)
                continue 
            end 

            # compute constants  
            const1 = dot(Ωinv, V[j]) # const1 = tr(Ωinv * V[j])
            mul!(w, V[j], v)
            const2 = dot(w, v) # const2 = y' * Ωinv * V[j] * Ωinv * y

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && penwt[j] > 0
              # set variance component to zero if weight = Inf 
              if isinf(penwt[j])
                  σ2[j] = zero(T)
              else
                  penstrength = λ * penwt[j]
                  # L1 penalty 
                  if isa(penfun, L1Penalty)  
                        σ2[j] = σ2[j] * √(const2 / (const1 + penstrength / sqrt(σ2[j])))
                  # MCP penalty 
                  elseif isa(penfun, MCPPenalty) 
                    if σ2[j] <= (penfun.γ * λ)^2
                        σ2[j] = σ2[j] * 
                            √(const2 / (const1 + (λ / sqrt(σ2[j])) - (1 / penfun.γ)))
                    else 
                        σ2[j] = σ2[j] * √(const2 / const1)  
                    end 
                  end 
              end

            # update variance component under no penalty 
            elseif isa(penfun, NoPenalty)
                σ2[j] = σ2[j] * √(const2 / const1)
            end

            # move onto the next variance component if estimated variance component is 0
            if iszero(σ2[j]) 
                continue 
            end 
            Ω .+= σ2[j] .* V[j]
        end # end of for loop over j

        # update last variance component  
        σ2[m + 1] = σ2[m + 1] * √(dot(v, v) / tr(Ωinv))
        # update diagonal entry of Ω
        for i in 1:n 
            Ω[i, i] += σ2[m + 1]
        end 
        # update Ωchol, Ωinv, v 
        Ωchol = cholesky!(Symmetric(Ω))
        #Ωchol = cholesky(Symmetric(Ω))
        Ωinv[:]  = inv(Ωchol)
        mul!(v, Ωinv, y)

        # update objective value 
        objold = obj
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)
        pen = 0.0
        for j in 1:m
            pen += penwt[j] * value(penfun, √σ2[j])
        end
        obj += loglConst + λ * pen

        # display current iterate if specified 
        if verbose
            println("iter = ", iter)
            println("σ2   = ", σ2)
            println("obj  = ", obj)
        end

        # check convergence
        if abs(obj - objold) < tol * (abs(objold) + 1)
            niters = iter
            break
        end

    end # end of iteration loop

    # output
    if niters == 0
      niters = maxiter
    end

    return σ2, obj, niters; #, Ω, Ωinv;
end


"""
    vcselectpath(y, X, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Project `y` to null space of `X` and generate solution path of variance components 
along varying lambda values.

# Input  
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
    note that V[end] should be identity matrix

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nlambda`: number of tuning parameter values, default is 100. 
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid.
- `σ2`: initial estimates.
- `maxiter`: maximum number of iteration for MM loop.
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output 
- `σ2path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
"""
function vcselectpath(
    y       :: AbstractVector{T},
    X       :: AbstractMatrix{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = T[],
    σ2      :: AbstractVector{T} = ones(T,length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}

    
    ## generate solution path based on penalty 
    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            ynew, Vnew = nullprojection(y, X, V)
            maxλ = maxlambda(ynew, Vnew; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        else # if lambda grid specified, make sure nlambda matches 
            nlambda = length(λpath)
        end 

        # no. groups 
        m = length(V) - 1

        # initialize solution path 
        σ2path = zeros(T, m + 1, nlambda)
        objpath = zeros(T, nlambda)
        niterspath = zeros(Int, nlambda)
        betapath = zeros(T, size(X, 2), nlambda)

        # create solution path 
        for iter in 1:length(λpath)
            λ = λpath[iter]
            σ2path[:, iter], betapath[:, iter], objpath[iter], niterspath[iter], = 
                vcselect(y, X, V; penfun=penfun, λ=λ, penwt=penwt, σ2=σ2, maxiter=maxiter, 
                tol=tol, verbose=verbose)
        end

    else # if no penalty, there is no lambda grid 
        σ2path, betapath, objpath, niterspath, = vcselect(y, X, V; penfun=penfun)
    end 

    # output 
    return σ2path, betapath, objpath, λpath, niterspath 


end 



"""
    vcselectpath(y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Generate solution path of variance components along varying lambda values.

# Input
- `y`: response vector
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
        note that V[end] should be identity matrix

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0).
- `nlambda`: number of tuning parameter values, default is 100. 
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid.
- `σ2`: initial estimates.
- `maxiter`: maximum number of iteration for MM loop.
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output 
- `σ2path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
- `niterspath`: vector of no. of iterations to convergence 
"""
function vcselectpath(
    y       :: AbstractVector{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = ones(T, nlambda),
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}


    ## generate solution path based on penalty 
    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ = maxlambda(y, V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        else # if lambda grid specified, make sure nlambda matches 
            nlambda = length(λpath)
        end 

        # no. groups 
        m = length(V) - 1

        # initialize solution path 
        σ2path = zeros(T, m + 1, nlambda)
        objpath = zeros(T, nlambda)
        niterspath = zeros(Int, nlambda)

        # create solution path 
        for iter in 1:nlambda
            λ = λpath[iter]
            σ2path[:, iter], objpath[iter], niterspath[iter] = vcselect(y, V; 
                    penfun=penfun, λ=λ, penwt=penwt, σ2=σ2, maxiter=maxiter, 
                    tol=tol, verbose=verbose)
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, niterspath, = vcselect(y, V; penfun=penfun)
    end 

    # output 
    return σ2path, objpath, λpath, niterspath


end 