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
- `niter`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `Ωinv`: precision (inverse covariance) matrix evaluated at the estimated variance components
"""
function vcselect( 
    y       :: AbstractVector{T},
    X       :: AbstractMatrix{T},
    V       :: AbstractVector{AbstractMatrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = one(T),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}

    ynew, Vnew = projectToNullSpace(y, X, V)

    # call vcselect 
    σ2, obj, niters, Ω, Ωinv = vcselect(ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, σ2=σ2,
                                        maxiter=maxiter, tol=tol, verbose=verbose)

    return σ2, obj, niters, Ω, Ωinv;
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
- `niter`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `Ωinv`: precision (inverse covariance) matrix evaluated at the estimated variance components
"""
function vcselect( 
    y       :: AbstractVector{T},
    V       :: AbstractVector{AbstractMatrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = one(T),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real} 

    # initialize algorithm
    n = length(y)       # no. observations
    m = length(V) - 1   # no. variance components
    Ω = zeros(n, n)     # covariance matrix 
    for j in 1:(m + 1)
        Ω .+= σ2[j] .* V[j]
    end
  
    Ωchol = cholesky(Hermitian(Ω))
    Ωinv = inv(Ωchol) 
    v = Ωinv * y
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v) # objective value 
    pen = 0.0
    for j in 1:m
        pen += penwt[j] * value(penfun, √σ2[j])
    end
    loglConst = (1//2) * n * log(2π)
    obj += loglConst + λ * pen
    if verbose
        println("iter = 0")
        println("σ2   = ", σ2)
        println("obj  = ", obj)
    end


    # MM loop
    niters = 0
    for iter in 1:maxiter
        # update variance components
        fill!(Ω, 0.0)
        for j in 1:m
            # compute constants  
            const1 = dot(Ωinv, V[j]) # const1 = tr(Ωinv * V[j])
            const2 = dot(V[j] * v, v) # const2 = y' * Ωinv * V[j] * Ωinv * y

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && penwt[j] > 0.0
              # set variance component to zero if weight = Inf 
              if penwt[j] == Inf
                  σ2[j] = 0.0
              else
                  penstrength = λ * penwt[j]
                  # L1 penalty 
                  if isa(penfun, L1Penalty) 
                      if σ2[j] != 0 # check if previous iterate is not zero 
                        σ2[j] = σ2[j] * √(const2 / (const1 + penstrength / sqrt(σ2[j])))
                      end
                  # MCP penalty 
                  elseif isa(penfun, MCPPenalty) 
                    if σ2[j] != 0 # check if previous iterate is not zero 
                        if σ2[j] <= (penfun.γ * λ)^2
                            σ2[j] = σ2[j] * 
                                √(const2 / (const1 + (λ / sqrt(σ2[j])) - (1 / penfun.γ)))
                        else 
                            σ2[j] = σ2[j] * √(const2 / const1)  
                        end 
                    end 
                  end 
              end
            # update variance component under no penalty 
            elseif isa(penfun, NoPenalty)
                σ2[j] = σ2[j] * √(const2 / const1)
            end
            Ω .+= σ2[j] .* V[j]
        end # end of for loop over j

        # update covariance matrix 
        σ2[m + 1] = σ2[m + 1] * √(dot(v, v) / tr(Ωinv))
        Ω[diagind(Ω)] .+= σ2[m + 1]
        Ωchol = cholesky(Hermitian(Ω))
        Ωinv  = inv(Ωchol)
        v = Ωinv * y

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

    return σ2, obj, niters, Ω, Ωinv;
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
    V       :: AbstractVector{AbstractMatrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(length(V)-1, T); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = zeros(T, nlambda),
    σ2      :: AbstractVector{T} = ones(length(V), T),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-6,
    verbose :: Bool = false
    ) where {T <: Real}

    # project response vector and covariance matrices 
    ynew, Vnew = projectToNullSpace(y, X, V)

    # call vcselectPath function 
    σ2path, objpath, λpath = vcselectpath(ynew, Vnew; penfun=penfun, penwt=penwt, 
                            nlambda=nlambda, λpath=λpath, σ2=σ2, maxiter=maxiter, tol=tol)

    # output 
    return σ2path, objpath, λpath 


end 



"""
    vcselectpath(y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Generate solution path of variance components along varying lambda values.

# Input
- `y`: response vector
- `X`: covariate matrix 
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
"""
function vcselectpath(
    y       :: AbstractVector{T},
    V       :: AbstractVector{AbstractMatrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(length(V)-1, T); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = zeros(T, nlambda),
    σ2      :: AbstractVector{T} = ones(length(V), T),
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
        end 

        # no. groups 
        m = length(V) - 1

        # initialize solution path 
        σ2path = zeros(m + 1, nlambda)
        objpath = zeros(nlambda)

        # create solution path 
        for iter in 1:length(λpath)
            λ = λpath[iter]
            σ2path[:, iter], objpath[iter], = vcselect(y, V; penfun=penfun, λ=λ, 
                        penwt=penwt, σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, = vcselect(y, V; penfun=penfun)
    end 

    # output 
    return σ2path, objpath, λpath 


end 