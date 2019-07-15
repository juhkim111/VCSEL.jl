"""
    vcselect(y, X, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Project covariate matrix `X` to null space of `X'` and 
call `vcselect(y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)`

# Input
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
    note (1) V[end] should be identity matrix or identity matrix divided by √n
    note (2) each V[i] needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each V[i] by its frobenius norm  

# Keyword
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty()
        γ for MCPPenalty() is 2.0 by default,
        if different γ (say 3.0) is desired, simply do MCPPenalty(3.0)
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `σ2`: initial values, default is (1,1,...,1)
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-8
- `verbose`: display switch, default is false 
- `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
    Default is true 

# Output
- `σ2`: vector of estimated variance components 
- `beta`: estimated fixed effects parameter, using REML estimates
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
"""
function vcselect( 
    y             :: AbstractVector{T},
    X             :: AbstractMatrix{T},
    V             :: AbstractVector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2            :: AbstractVector{T} = ones(T, length(V)),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-8,
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true
    ) where {T <: Real}

    if checkfrobnorm 
        # check frob norm equals to 1 
        checkfrobnorm!(V)
    end 

    # project onto null space 
    ynew, Vnew = nullprojection(y, X, V)

    # call vcselect 
    σ2, obj, niters, = vcselect(ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                            σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose,
                            checkfrobnorm=false)

    # update Ω with estimated variance components 
    Ω = zeros(T, size(V[1]))
    for i in eachindex(σ2)
        if iszero(σ2[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    # estimate fixed effects 
    β = getfixedeffects(y, X, Ω)

    return σ2, β, obj, niters, Ω;
end

"""
    vcselect(y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]`, `V[end] = I` and `n` is the length of `y`.
Minimization is achieved via majorization-minimization (MM) algorithm. 

# Input
- `y`: response vector
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default

# Keyword
- `penfun`: penalty function, e.g., `NoPenalty()` (default), `L1Penalty()`, `MCPPenalty(γ = 2.0)`
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights where penwt[end] must equal to 0, 
        default is (1,1,...,1,0)
- `σ2`: initial values, default is (1,1,...,1)
- `Ω`: initial overall covariance matrix `Ω`
- `Ωinv`: initial inverse matrix of overall covariance matrix `Ω`
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-8
- `verbose`: display switch, default is false 
- `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
        default is true 

# Output
- `σ2`: vector of estimated variance components 
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `objvec`: vector of objective values at each iteration,
        returned only if `verbose` is true
"""
function vcselect( 
    y             :: AbstractVector{T},
    V             :: Vector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2            :: AbstractVector{T} = ones(T, length(V)),
    Ω             :: AbstractMatrix{T} = zeros(T, size(V[1])), 
    Ωinv          :: AbstractMatrix{T} = zeros(T, size(V[1])),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-8,
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true
    ) where {T <: Real} 

    # check frob norm equals to 1 
    if checkfrobnorm 
        checkfrobnorm!(V)
    end 

    # initialize algorithm
    n = length(y)       # no. observations
    m = length(V) - 1   # no. variance components
    Ω = fill!(Ω, 0)     # covariance matrix 
    for j in 1:length(V)
        Ω .+= σ2[j] .* V[j]
    end
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    v = Ωinv * y
    w = similar(v) 
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v) # objective value 
    pen = 0.0
    for j in 1:m
        pen += penwt[j] * value(penfun, √σ2[j])
    end
    loglConst = (1//2) * n * log(2π) 
    obj += loglConst + λ * pen

    # display 
    if verbose
        println("iter = 0")
        println("σ2   = ", σ2)
        println("obj  = ", obj)
        #objvec = obj
    end    

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

        # update last variance component and Ω
        σ2[end] = σ2[end] * √(dot(v, v) / tr(Ωinv))
        Ω .+= σ2[end] .* V[end]

        # update Ωchol, Ωinv, v 
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(v, Ωinv, y)

        # update objective value 
        objold = obj
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v) 
        if !isa(penfun, NoPenalty)
            pen = 0.0
            for j in 1:m
                if σ2[j] == 0
                    continue
                else 
                    pen += penwt[j] * value(penfun, √σ2[j])
                end 
            end
            obj += loglConst + λ * pen
        else 
            obj += loglConst 
        end

        # display current iterate if specified 
        # if verbose && iter == 1
        #     println("iter = ", iter)
        #     println("σ2   = ", σ2)
        #     println("obj  = ", obj)
        #     #objvec = [objvec; obj] 
        # end

        # check convergence
        if abs(obj - objold) < tol * (abs(obj) + 1)
            niters = iter
            break
        end

    end # end of iteration loop

    # construct Ω matrix 
    fill!(Ω, 0)
    for i in eachindex(σ2)
        if iszero(σ2[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    # output
    if niters == 0
      niters = maxiter
    end

    if verbose 
        return σ2, obj, niters, Ω; #, objvec;
    else 
        return σ2, obj, niters, Ω;
    end
end
"""
    vcselectpath(y, X, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-8)

Project `y` to null space of `X` and generate solution path of variance components 
along varying lambda values.

# Input  
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nlambda`: number of tuning parameter values, default is 100. 
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid.
- `σ2`: initial estimates.
- `maxiter`: maximum number of iteration for MM loop.
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-8
- `verbose`: display switch, default is false 
- `fixedeffects`: whether user wants fixed effects parameter 
        to be estimated and returned, default is false 

# Output 
- `σ2path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
- `niterspath`: vector of no. of iterations to convergence 
- `βpath`: matrix of estimated fixed effects at each tuning parameter `λ`
"""
function vcselectpath(
    y            :: AbstractVector{T},
    X            :: AbstractMatrix{T},
    V            :: AbstractVector{Matrix{T}};
    penfun       :: Penalty = NoPenalty(),
    penwt        :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda      :: Int = 100, 
    λpath        :: AbstractVector{T} = T[],
    σ2           :: AbstractVector{T} = ones(T, length(V)),
    maxiter      :: Int = 1000,
    tol          :: AbstractFloat = 1e-8,
    verbose      :: Bool = false,
    fixedeffects :: Bool = false 
    ) where {T <: Real}

    # project y and V onto nullspace of X
    ynew, Vnew = nullprojection(y, X, V)

    # 
    σ2path, objpath, λpath, niterspath = vcselectpath(ynew, Vnew;
        penfun=penfun, penwt=penwt, nlambda=nlambda, λpath=λpath, σ2=σ2, maxiter=maxiter,
        tol=tol, verbose=verbose)

    # if user wants fixed effects estimates, estimate β
    if fixedeffects 
        βpath = zeros(T, size(X, 2), nlambda)
        for iter in 1:length(λpath)
            βpath[:, iter] = getfixedeffects(y, X, V, view(σ2path, :, iter))
        end 
        # output 
        return σ2path, objpath, λpath, niterspath, βpath
    else 
        return σ2path, objpath, λpath, niterspath
    end 
end 
"""
    vcselectpath(y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-8)

Generate solution path of variance components along varying lambda values.

# Input
- `y`: response vector
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0).
- `nlambda`: number of tuning parameter values, default is 100. 
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid.
- `σ2`: initial estimates.
- `maxiter`: maximum number of iteration for MM loop.
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-8
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
    λpath   :: AbstractVector{T} = T[],
    σ2      :: AbstractVector = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-8,
    verbose :: Bool = false
    ) where {T <: Real}

    #checkfrobnorm!(V)

    ## generate solution path based on penalty 
    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ, = maxlambda(y, V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        else # if lambda grid specified, make sure nlambda matches 
            nlambda = length(λpath)
        end 

        # initialize arrays 
        σ2path = zeros(T, length(V), nlambda)
        objpath = zeros(T, nlambda)
        niterspath = zeros(Int, nlambda)

        # create solution path 
        for iter in 1:nlambda 
            σ2, objpath[iter], niterspath[iter], = 
                    vcselect(y, V; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                    σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose, checkfrobnorm=false)
            σ2path[:, iter] = σ2 
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, niterspath, = vcselect(y, V; 
                penfun=penfun, maxiter=maxiter, tol=tol, verbose=verbose, 
                checkfrobnorm=false)
    end 

    # output 
    return σ2path, objpath, λpath, niterspath

end 

"""
    vcselectpath!(vcm; penfun, penwt, nλ, λpath, maxiters, tol, verbose, fixedeffects)

Generate solution path of variance components along varying lambda values.

# Input
- `vcm`: VCModel

# Keyword 
- `penfun`: penalty function, default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nlambda`: number of tuning parameter values, default is 100
- `λpath`: user-supplied grid of tuning parameter values
        If unspeficied, internally generate a grid
- `σ2`: initial estimates, default is (1,1,...,1)
- `maxiter`: maximum number of iteration for MM loop, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-8
- `verbose`: display switch, default is false 

# Output 
- `σ2path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `objpath`: vector of objective values at each tuning parameter `λ`
- `λpath`: vector of tuning parameter values used 
- `niterspath`: vector of no. of iterations to convergence 

"""
function vcselectpath!(
    vcm          :: VCModel; 
    penfun       :: Penalty = NoPenalty(), 
    penwt        :: AbstractArray = [ones(nvarcomps(vcm)-1); 0],
    nλ           :: Int = 100, 
    λpath        :: AbstractArray = zeros(0), 
    maxiters     :: Int = 1000, 
    tol          :: AbstractFloat = 1e-8, 
    verbose      :: Bool = false #, 
    #fixedeffects :: Bool = false
    ) 

    # handle errors 
    @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty(), L1Penalty(), or MCPPenalty()!\n"
    @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
    @assert size(λpath, 2) <= 1 "λpath must be one-dimensional array!\n"
    @assert maxiters > 0 "maxiters should be a positive integer!\n"

    # dimension of X 
    p = ncovariates(vcm)
    d = length(vcm)

    # type 
    T = eltype(vcm.Y)

    ## generate solution path based on penalty 
    if penfun != NoPenalty()

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ, = maxlambda(vcm.Y, vcm.V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nλ)
        else # if lambda grid specified, make sure nlambda matches 
            nλ = length(λpath)
        end 

        # initialize arrays 
        objpath = zeros(T, nλ)
        niterspath = zeros(Int, nλ)
        if length(vcm) == 1
            # initialize arrays
            Σ̂path = zeros(T, nvarcomps(vcm), nλ)
            β̂path = zeros(T, p, nλ)
            # solution path 
            for iter in 1:nλ
                _, _, objpath[iter], niterspath[iter], = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
                Σ̂path[:, iter] = vcm.Σ 
                β̂path[:, iter] = vcm.β
            end
        elseif length(vcm) > 1
            # initialize arrayss
            Σ̂path = fill(Matrix{Float64}(undef, d, d), nvarcomps(vcm), nλ)
            β̂path = fill(Matrix{Float64}(undef, p, d), nλ)
            # solution path 
            for iter in 1:nλ
                _, _, objpath[iter], niterspath[iter], = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
                Σ̂path[:, iter] = vcm.Σ 
                β̂path[iter] = vcm.β
            end
        end 

    else # if no penalty, there is no lambda grid 
        Σ̂path, β̂path, objpath, niterspath, = vcselect!(vcm; penfun=penfun, penwt=penwt, 
            maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
    end 
   
    return Σ̂path, β̂path, λpath, objpath, niterspath

end 


"""
    vcselect!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

# Input 
- `vcm`: VCModel

# Keyword Argument 
- `penfun`: penalty function, default is NoPenalty()
- `λ`: tuning parameter, default is 1      
- `penwt`: penalty weights, default is [1,...1,0]
- `maxiters`: maximum number of iterations, default is 1000
- `tol`: tolerance for convergence, default is 1e-8
- `verbose`: display switch, default is false  
- `checktype`: check argument type switch, default is true

# Output 
- `Σ̂`: estimated variance components 
- `β̂`: estimated fixed effects parameters vector 
- `obj`: objective value at convergence 
- `niters`: number of iterations to convergence 
- `Ω̂`: covariance matrix at estimated variance components 
"""
function vcselect!(
    vcm       :: VCModel;
    penfun    :: Penalty = NoPenalty(),
    λ         :: Real = 1.0,
    penwt     :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    maxiters  :: Int = 1000,
    tol       :: Real = 1e-8,
    verbose   :: Bool = false,
    checktype :: Bool = true 
    ) 

    # handle errors 
    if checktype 
        @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty(), L1Penalty(), or MCPPenalty()!\n"
        @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
        @assert maxiters > 0 "maxiters should be a positive integer!\n"
        # start point has to be strictly positive
        @assert all(norm.(vcm.Σ) .> 0) "starting Σ should be strictly positive or non-zero matrices"
    end 

    # univariate update 
    if length(vcm) == 1
        if verbose 
            Σ̂, β̂, obj, niters, Ω̂, objvec = mm_update_σ2!(vcm; penfun=penfun, λ=λ, 
                penwt=penwt, maxiters=maxiters, tol=tol, verbose=true)
            return Σ̂, β̂, obj, niters, Ω̂, objvec
        else 
            Σ̂, β̂, obj, niters, Ω̂ = mm_update_σ2!(vcm; penfun=penfun, λ=λ, 
                penwt=penwt, maxiters=maxiters, tol=tol)
            return Σ̂, β̂, obj, niters, Ω̂
        end 
      
    # multivariate update 
    elseif length(vcm) > 1
        if verbose 
            Σ̂, β̂, obj, niters, Ω̂, objvec = mm_update_Σ!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
            # output 
            return Σ̂, β̂, obj, niters, Ω̂, objvec
        else 
            Σ̂, β̂, obj, niters, Ω̂ = mm_update_Σ!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
            # output 
            return Σ̂, β̂, obj, niters, Ω̂
        end 
    end 

end 

"""
    mm_update_Σ!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `Σ` using MM algorithm.
"""
function mm_update_Σ!(
    vcm       :: VCModel;
    penfun    :: Penalty = NoPenalty(),
    λ         :: Real = 1.0,
    penwt     :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    maxiters  :: Int = 1000,
    tol       :: Real = 1e-8,
    verbose   :: Bool = false 
    ) 

    # initialize algorithm 
    n, d = size(vcm)
    m = nvarcomps(vcm) 

    # working arrays 
    kron_I_one = kron(Matrix(I, d, d), ones(n)) # dn x d
    # ones_d = ones(d, d)
    # for i in 1:m
    #     #vcm.kron_ones_V[i] = kron(ones_d, vcm.V[i])
    # end 

    # initial objective value 
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)

    # # display 
    if verbose 
        println("iter = 0")
        #println("Σ    = ", vcm.Σ)
        println("obj  = ", obj)
        objvec = obj 
    end  

    ## MM loop 
    niters = 0
    for iter in 1:maxiters
        for i in 1:m 
            # if previous iterate is zero, move on to the next component
            if iszero(norm(vcm.Σ[i])) 
                continue 
            # if penalty weight is zero, move onto the next component 
            elseif iszero(penwt[i]) && i < m
                fill!(vcm.Σ[i], 0)
                #vcm.Σ[i] = zeros(d, d)
                continue 
            end 

            # `(kron_I_one)' * [kron(ones(d, d), V[i]) .* Ωinv] * (kron_I_one)`
            copyto!(vcm.Mndxnd, vcm.kron_ones_V[i] .* vcm.Ωinv)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', kron_I_one, vcm.Mndxnd * kron_I_one))
            #vcm.Mndxnd = vcm.kron_ones_V[i] .* vcm.Ωinv 
            #vcm.Mdxd = BLAS.gemm('T', 'N', kron_I_one, vcm.Mndxnd * kron_I_one) # d x d

            # add penalty unless it's the last variance component 
            if isa(penfun, L1Penalty) && i < m 
                penconst = λ * penwt[i] / √tr(vcm.Σ[i])
                for j in 1:d
                    vcm.Mdxd[j, j] += penconst  
                end             
            end 

            #vcm.L = cholesky!(Symmetric(vcm.Mdxd)).L
            copyto!(vcm.L, cholesky!(Symmetric(vcm.Mdxd)).L)
            vcm.Linv[:] = inv(vcm.L)

            # 
            copyto!(vcm.Mdxd, vcm.Σ[i] * vcm.L)
            copyto!(vcm.Mdxd, vcm.R * vcm.Mdxd)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', vcm.Mnxd, vcm.V[i] * vcm.Mnxd))
            # vcm.Mdxd = vcm.Σ[i] * vcm.L
            # vcm.Mnxd = vcm.R * vcm.Mdxd # n x d
            # vcm.Mdxd = BLAS.gemm('T', 'N', vcm.Mnxd, vcm.V[i] * vcm.Mnxd)

            # 
            storage = eigen!(Symmetric(vcm.Mdxd))
            # if negative value, set it to 0
            @inbounds for k in 1:d
                storage.values[k] = storage.values[k] > 0 ? √storage.values[k] : 0
            end 
            copyto!(vcm.Σ[i], BLAS.gemm('N', 'T', storage.vectors * Diagonal(storage.values), storage.vectors))
            copyto!(vcm.Σ[i], BLAS.gemm('T', 'N', vcm.Linv, vcm.Σ[i] * vcm.Linv))
            # vcm.Σ[i] = BLAS.gemm('N', 'T', storage.vectors * Diagonal(storage.values), storage.vectors)
            # vcm.Σ[i] = BLAS.gemm('T', 'N', vcm.Linv, vcm.Σ[i] * vcm.Linv)

        end 

        clamp_diagonal!(vcm.Σ[end], tol, Inf)

        # update working arrays 
        updateΩ!(vcm)
        update_arrays!(vcm)

        # update objective value 
        objold = obj 
        obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
        
        # display 
        if verbose 
            println("iter = ", iter)
            #println("Σ    = ", vcm.Σ)
            println("obj  = ", obj)
            objvec = [objvec; obj]
        end

        # check convergence 
        if abs(obj - objold) < tol * (abs(obj) + 1)
            niters = iter 
            break 
        end 

    end # end of iteration 

    # construct final Ω matrix
    updateΩ!(vcm)
    updateΩobs!(vcm)
    updateβ!(vcm)

    # output 
    if niters == 0 
        niters = maxiters
    end 
 
    if verbose  
        return vcm.Σ, vcm.β, obj, niters, vcm.Ωobs, objvec; 
    else 
        return vcm.Σ, vcm.β, obj, niters, vcm.Ωobs; 
    end 
end 

"""
    mm_update_σ2!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `σ2` using MM algorithm. 
"""
function mm_update_σ2!(
    vcm :: VCModel;
    penfun    :: Penalty = NoPenalty(),
    λ         :: Real = 1.0,
    penwt     :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    maxiters  :: Int = 1000,
    tol       :: Real = 1e-8,
    verbose   :: Bool = false 
    )

    # initialize algorithm 
    n = size(vcm)[1]
    m = nvarcomps(vcm)

    # initial objective value 
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
  
    # display 
    if verbose
        println("iter = 0")
        println("σ2   = ", vcm.Σ)
        println("obj  = ", obj)
        objvec = obj 
    end    

    σ2tmp = zeros(m)
  
    # MM loop 
    niters = 0
    for iter in 1:maxiters
          # update variance components
          for j in 1:(m - 1)
              # move onto the next variance component if previous iterate is 0
              if iszero(vcm.Σ[j]) 
                  σ2tmp[j] = 0 
                  continue 
              # set to 0 and move onto the next variance component if penalty weight is 0
              elseif iszero(penwt[j]) 
                  #vcm.Σ[j] = 0
                  σ2tmp[j] = 0 
                  #fill!(vcm.Σ[j], 0)
                  continue 
              end 
  
              # compute constants  
              const1 = dot(vcm.Ωinv, vcm.V[j]) # const1 = tr(Ωinv * V[j])
              mul!(vcm.Mnxd, vcm.V[j], vcm.ΩinvY)
              const2 = dot(vcm.Mnxd, vcm.ΩinvY) # const2 = y' * Ωinv * V[j] * Ωinv * y
  
              # update variance component under specified penalty 
              if !isa(penfun, NoPenalty) 
                    penstrength = λ * penwt[j]
                    # L1 penalty 
                    if isa(penfun, L1Penalty)  
                        σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + penstrength / sqrt(vcm.Σ[j])))
                        #vcm.Σ[j] *= √(const2 / (const1 + penstrength / sqrt(vcm.Σ[j])))
                    # MCP penalty 
                    elseif isa(penfun, MCPPenalty) 
                        if vcm.Σ[j] <= (penfun.γ * λ)^2
                            σ2tmp[j] = vcm.Σ[j] *
                                √(const2 / (const1 + (λ / sqrt(vcm.Σ[j])) - (1 / penfun.γ)))
                        
                            #vcm.Σ[j] *= 
                            #√(const2 / (const1 + (λ / sqrt(vcm.Σ[j])) - (1 / penfun.γ)))
                        else 
                            
                            σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
                            #vcm.Σ[j] *= √(const2 / const1)  
                        end 
                    end 
              # update variance component under no penalty 
              elseif isa(penfun, NoPenalty)
                σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
                #vcm.Σ[j] *= √(const2 / const1)
              end
  
          end # end of for loop over j

          # update last variance component and Ω
          σ2tmp[end] = vcm.Σ[end] *  √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
          σ2tmp[end] = clamp(σ2tmp[end], tol, Inf)
        #   vcm.Σ[end] *= √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
        #   clamp!(vcm.Σ[end], tol, Inf)
          
        #   vcm.Σ[end] *= √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
        #   vcm.Σ[end] = clamp(vcm.Σ[end], tol, Inf)

          vcm.Σ .= σ2tmp

          # update working arrays 
          updateΩ!(vcm)
          update_arrays!(vcm)
  
          # update objective value 
          objold = obj
          obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
  
          # display current iterate if specified 
          if verbose
              println("iter = ", iter)
              println("σ2   = ", vcm.Σ)
              println("obj  = ", obj)
              objvec = [objvec; obj] 
          end
  
          # check convergence
          if abs(obj - objold) < tol * (abs(obj) + 1)
              niters = iter
              break
          end
  
      end # end of iteration 
  
      # construct Ω matrix 
      updateΩ!(vcm)
      updateΩobs!(vcm)
      updateβ!(vcm)
  
      # output
      if niters == 0
        niters = maxiters
      end
   
      # 
      if verbose 
        return vcm.Σ, vcm.β, obj, niters, vcm.Ωobs, objvec; 
      else
        return vcm.Σ, vcm.β, obj, niters, vcm.Ωobs; 
      end 

end 