"""
    vcselectpath!(vcm; penfun=NoPenalty(), penwt=[ones(nvarcomps(vcm)-1); 0], nλ=100, 
                    λpath=zeros(0), maxiters=1000, tol=1e-6, verbose=false)

Generate solution path of variance components along varying lambda values.

# Input
<<<<<<< HEAD
- `y`: response vector
- `X`: covariate vector or matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I/√n)
    note that each V[i] needs to have frobenius norm 1, and that V[end] should be 
    identity matrix divided by √n

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

# Output
- `σ2`: vector of estimated variance components 
- `β`: estimated fixed effects parameter, using REML estimates
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
"""
function vcselect( 
    y             :: AbstractVector{T},
    X             :: AbstractVecOrMat{T},
    V             :: AbstractVector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    penwt         :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    σ2            :: AbstractVector{T} = ones(T, length(V)),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-8,
    verbose       :: Bool = false
    ) where {T <: Real}

    # project onto null space 
    ynew, Vnew = nullprojection(y, X, V)

    # call vcselect 
    if verbose 
        σ2, obj, niters, _, objvec  = vcselect(ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)
    else 
        σ2, obj, niters,  = vcselect(ynew, Vnew; penfun=penfun, λ=λ, penwt=penwt, 
                σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)
    end 
    
    # update Ω with estimated variance components 
    Ω = zeros(T, size(V[1]))
    for i in eachindex(σ2)
        if iszero(σ2[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
=======
- `vcm`: [`VCModel`](@ref).

# Keyword 
- `penfun`: penalty function (e.g. `NoPenalty()``, `L1Penalty()``, `MCPPenalty()`). 
        Default is `NoPenalty()`.
- `penwt`: weights for penalty term. Default is (1,1,...1,0).
- `nλ`: number of tuning parameter values. Default is 100.
- `λpath`: a user supplied `λ` sequence in ascending order. Typically the program computes its own `λ` 
        sequence based on `nλ`; supplying `λpath` overrides this.
- `maxiter`: maximum number of iteration for MM loop. Default is 1000.
- `standardize`: logical flag for covariance matrix standardization. Default is `true`.
    If true, `V[i]` is standardized by its Frobenius norm.
- `tol`: convergence tolerance. Default is `1e-6`.

# Output 
- `Σ̂path`: matrix of estimated variance components at each tuning parameter `λ`.
        Each column gives vector of estimated variance components `σ2` at certain `λ`.
- `β̂path`: matrix of fixed parameter estimates at each `λ`.
- `λpath`: sequence of `λ` values used.
- `objpath`: vector of objective values at each tuning parameter `λ`.
- `niterspath`: vector of no. of iterations to convergence.
"""
function vcselectpath!(
    vcm          :: VCModel; 
    penfun       :: Penalty = NoPenalty(), 
    penwt        :: AbstractArray = [ones(nvarcomps(vcm)-1); 0],
    nλ           :: Int = 100, 
    λpath        :: AbstractArray = zeros(0), 
    maxiters     :: Int = 1000, 
    standardize  :: Bool = true, 
    tol          :: AbstractFloat = 1e-6
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
        if typeof(vcm.Σ[1]) <: Matrix 
            # initialize arrays
            Σ̂path = Array{Matrix{T}}(undef, nvarcomps(vcm), nλ)
            β̂path = [zeros(T, p, d) for i in 1:nλ]
            # solution path 
            for iter in 1:nλ
                _, objpath[iter], niterspath[iter], = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=false, checktype=false,
                        standardize=standardize)
                Σ̂path[:, iter] = vcm.Σ 
                β̂path[iter] .= vcm.β
            end
        else
            # initialize arrays
            Σ̂path = zeros(T, nvarcomps(vcm), nλ)
            β̂path = zeros(T, p, nλ)
            # solution path 
            for iter in 1:nλ
                _, objpath[iter], niterspath[iter], = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=false, checktype=false,
                        standardize=standardize)
                Σ̂path[:, iter] .= vcm.Σ
                β̂path[:, iter] .= vcm.β
            end
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
        end 

        return Σ̂path, β̂path, λpath, objpath, niterspath

<<<<<<< HEAD
    if verbose 
        return σ2, β, obj, niters, Ω, objvec;
    else 
        return σ2, β, obj, niters, Ω;
    end
end
=======
    else # if no penalty, there is no lambda grid 
        _, objpath, niterspath, = vcselect!(vcm; penfun=penfun, penwt=penwt, 
            maxiters=maxiters, tol=tol, verbose=false, checktype=false)
        
        return vcm.Σ, vcm.β, zeros(1), objpath, niterspath 
    end  
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

end 

"""
    vcselect!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

# Input 
- `vcm`: [`VCModel`](@ref).

# Keyword Argument 
- `penfun`: penalty function. Default is `NoPenalty()`.
- `λ`: tuning parameter. Default is 1.    
- `penwt`: penalty weights. Default is (1,...1,0).
- `standardize`: logical flag for covariance matrix standardization. Default is `true`.
    If true, `V[i]` is standardized by its Frobenius norm, and parameter estimates are 
    returned on the original scale.
- `maxiters`: maximum number of iterations. Default is 1000.
- `tol`: convergence tolerance. Default is `1e-6`.
- `verbose`: display switch. Default is false.
- `checktype`: check argument type switch. Default is true.

<<<<<<< HEAD
# Input
- `y`: response vector
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I/√n)
    note that each V[i] needs to have frobenius norm 1, and 
    that V[end] should be identity matrix divided by √n

# Keyword
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty(γ = 2.0)
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `σ2`: initial values, default is (1,1,...,1)
- `Ω`: initial overall covariance matrix `Ω`
- `Ωinv`: initial inverse matrix of overall covariance matrix `Ω`
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
- `verbose`: display switch, default is false 

# Output
- `σ2`: vector of estimated variance components 
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `objvec`: vector of objective values at each iteration 
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
    verbose       :: Bool = false
    ) where {T <: Real} 


    # 
    ϵ = convert(T, 1e-8)

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
    loglConst = (1//2) * n * log(2π) 
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v) # objective value  
    pen = 0.0
    for j in 1:m
        if iszero(σ2[j])
            continue 
        else 
            pen += penwt[j] * value(penfun, √σ2[j])
        end 
    end
    obj += loglConst + λ * pen
=======
# Output 
- `vcm`: VCModel with updated `Σ` and `β`.
- `obj`: objective value at convergence.
- `niters`: number of iterations to convergence.
- `objvec`: vector of objvective values at each iteration.
"""
function vcselect!(
    vcm          :: VCModel;
    penfun       :: Penalty = NoPenalty(),
    λ            :: Real = 1.0,
    penwt        :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize  :: Bool = false, 
    maxiters     :: Int = 1000,
    tol          :: Real = 1e-6,
    verbose      :: Bool = false,
    checktype    :: Bool = true 
    ) 

    # handle errors 
    if checktype 
        @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty(), L1Penalty(), or MCPPenalty()!\n"
        @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
        @assert maxiters > 0 "maxiters should be a positive integer!\n"
        # start point has to be strictly positive
        @assert all(norm.(vcm.Σ) .> 0) "starting Σ should be strictly positive or non-zero matrices"
    end 

    # update weight with reciprocal of frobenius norm 
    if standardize 
        vcm.wt .= 1 ./ norm.(vcm.V)
    end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

    # multivariate update 
    if typeof(vcm.Σ[1]) <: Matrix  # length(vcm) > 1
        _, obj, niters, objvec = mm_update_Σ!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
    
    # univariate update 
    else
        _, obj, niters, objvec = mm_update_σ2!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
    end 

    # output 
    return vcm, obj, niters, objvec

end 

"""
    mm_update_Σ!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `Σ` using MM algorithm.
"""
function mm_update_Σ!(
    vcm         :: VCModel;
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize :: Bool = false, 
    maxiters    :: Int = 1000, 
    tol         :: Real = 1e-6,
    verbose     :: Bool = false 
    ) 

    # initialize algorithm 
    n, d = size(vcm)

    # working arrays 
    kron_I_one = kron(Matrix(I, d, d), ones(n)) # dn x d
   
    # initial objective value 
    updateΩ!(vcm)
    update_arrays!(vcm)
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)

    # # display 
    if verbose 
        println("iter = 0")
        println("Σ    = ", vcm.Σ)
        println("obj  = ", obj)
        objvec = obj 
    end  

    Σtmp = deepcopy(vcm.Σ)

    ## MM loop 
    niters = 0
    for iter in 1:maxiters
        for i in eachindex(vcm.Σ)
            # if previous iterate is zero, move on to the next component
            if iszero(norm(vcm.Σ[i])) 
                continue 
            end 

<<<<<<< HEAD
            # compute constants  
            const1 = dot(Ωinv, V[j]) # const1 = tr(Ωinv * V[j])
            mul!(w, V[j], v)
            const2 = dot(w, v) # const2 = y' * Ωinv * V[j] * Ωinv * y

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && penwt[j] > 0
              # set variance component to zero if weight = Inf 
              if isinf(penwt[j])
                  σ2[j] = zero(T)
                  continue 
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
=======
            # `(kron_I_one)' * [kron(ones(d, d), V[i]) .* Ωinv] * (kron_I_one)`
            copyto!(vcm.Mndxnd, vcm.kron_ones_V[i] .* vcm.Ωinv)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', vcm.wt[i], kron_I_one, vcm.Mndxnd * kron_I_one))
        
            # add penalty as long as it is NOT the last variance component 
            if isa(penfun, L1Penalty) && i < nvarcomps(vcm) 
                penconst = λ * penwt[i] / √tr(vcm.Σ[i])
                for j in 1:d
                    vcm.Mdxd[j, j] += penconst  
                end             
            elseif isa(penfun, MCPPenalty) && (i < nvarcomps(vcm)) && (√tr(vcm.Σ[i]) <= penfun.γ * λ)
                penconst = λ * penwt[i] / √tr(vcm.Σ[i]) - 1 / penfun.γ
                for j in 1:d
                    vcm.Mdxd[j, j] += penconst  
                end
            end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

            # cholesky decomposition & inverse 
            copyto!(vcm.L, cholesky!(Symmetric(vcm.Mdxd)).L)
            vcm.Linv[:] = inv(vcm.L)

            # 
            copyto!(vcm.Mdxd, vcm.Σ[i] * vcm.L)
            copyto!(vcm.Mnxd, vcm.R * vcm.Mdxd)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', vcm.Mnxd, vcm.V[i] * vcm.Mnxd))

            # 
            storage = eigen!(Symmetric(vcm.Mdxd))
            # if negative value, set it to 0
            @inbounds for k in 1:d
                storage.values[k] = storage.values[k] > 0 ? √storage.values[k] : 0
            end 
<<<<<<< HEAD
            Ω .+= σ2[j] .* V[j]
        end # end of for loop over j

        # update last variance component  
        σ2[end] = σ2[end] * √(dot(v, v) / tr(Ωinv))
        σ2[end] = clamp(σ2[end], ϵ, T(Inf))

        # update overall covariance matrix 
        Ω .+= σ2[end] .* V[end]

        # update Ωchol, Ωinv, v 
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(v, Ωinv, y)

        # update objective value 
        objold = obj
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)
        pen = 0.0
        for j in 1:m
            if iszero(σ2[j])
                continue 
            else 
                pen += penwt[j] * value(penfun, √σ2[j])
            end
        end
        obj += loglConst + λ * pen
=======
            copyto!(Σtmp[i], BLAS.gemm('N', 'T', 
                    storage.vectors * Diagonal(storage.values), storage.vectors))
            copyto!(Σtmp[i], BLAS.gemm('T', 'N', 
                    sqrt(vcm.wt[i]), vcm.Linv, Σtmp[i] * vcm.Linv))
        end 

        # update Σ
        clamp_diagonal!(Σtmp[end], tol, Inf)
        vcm.Σ .= Σtmp

        # update working arrays 
        updateΩ!(vcm)
        update_arrays!(vcm)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

        # update objective value 
        objold = obj 
        obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
        
        # display 
        if verbose 
            println("iter = ", iter)
            println("Σ    = ", vcm.Σ)
            println("obj  = ", obj)
            objvec = [objvec; obj]
        end

        # check convergence 
        if abs(obj - objold) < tol * (abs(obj) + 1)
            niters = iter 
            break 
        end 

    end # end of iteration 

    # back to original scale  
    if standardize 
        vcm.Σ .*= vcm.wt
        vcm.wt .= ones(nvarcomps(vcm))
    end 

    # construct final Ω matrix
    updateΩ!(vcm)
    updateΩest!(vcm)
    updateβ!(vcm)

    # output 
    if niters == 0 
        niters = maxiters
    end 
 
    if verbose 
<<<<<<< HEAD
        return σ2, obj, niters, Ω, objvec;
    else 
        return σ2, obj, niters, Ω;
    end
end
"""
    vcselectpath(y, X, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(V)), maxiter=1000, tol=1e-6)

Project `y` to null space of `X` and generate solution path of variance components 
along varying lambda values.

# Input  
- `y`: response vector
- `X`: covariate vector or matrix 
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
- `fixedeffects`: whether user wants fixed effects parameter 
        to be estimated and returned, default is false 
=======
        return vcm, obj, niters, objvec
    else
        return vcm, obj, niters, zeros(0)
    end 
end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

"""
<<<<<<< HEAD
function vcselectpath(
    y            :: AbstractVector{T},
    X            :: AbstractVecOrMat{T},
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
        p = size(X, 2)
        if p > 1
            βpath = zeros(T, p, nlambda)
            for iter in 1:length(λpath)
                βpath[:, iter] = betaestimate(y, X, V, view(σ2path, :, iter))
            end 
        elseif p == 1
            βpath = zeros(T, nlambda)
            for iter in 1:length(λpath)
                βpath[iter] = betaestimate(y, X, V, view(σ2path, :, iter))
            end 
        end 
        # output 
        return σ2path, objpath, λpath, niterspath, βpath
    else 
        return σ2path, objpath, λpath, niterspath
    end 
end 
=======
    mm_update_σ2!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `σ2` using MM algorithm. 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
"""
function mm_update_σ2!(
    vcm         :: VCModel;
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize :: Bool = false, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-6,
    verbose     :: Bool = false 
    )

    # initialize algorithm 
    n = size(vcm)[1]
    m = nvarcomps(vcm) - 1

    # initial objective value 
    updateΩ!(vcm)
    update_arrays!(vcm)
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
  
    # display 
    if verbose
        println("iter = 0")
        println("σ2   = ", vcm.Σ)
        println("obj  = ", obj)
        objvec = obj 
    end    

    σ2tmp = zeros(m + 1)
  
    # MM loop 
    niters = 0
    for iter in 1:maxiters
          # update variance components
          for j in 1:m
              # move onto the next variance component if previous iterate is 0
              if iszero(vcm.Σ[j]) 
                  σ2tmp[j] = 0 
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
                        σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + 
                                penstrength / (vcm.wt[j] * sqrt(vcm.Σ[j]))))
                    # MCP penalty 
                    elseif isa(penfun, MCPPenalty) 
                        if vcm.Σ[j] <= (penfun.γ * λ)^2
                            σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + 
                                (λ / sqrt(vcm.Σ[j]) - 1 / penfun.γ) * (1 / vcm.wt[j])))
                        else 
                            σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
                        end 
                    end 
              # update variance component under no penalty 
              elseif isa(penfun, NoPenalty)
                σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
              end
  
          end # end of for loop over j

          # update last variance component and Ω
          σ2tmp[end] = vcm.Σ[end] *  √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
          σ2tmp[end] = clamp(σ2tmp[end], tol, Inf)

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

      # back to original scale  
      if standardize 
            vcm.Σ .*= vcm.wt
            vcm.wt .= ones(m + 1)
      end 
  
      # construct Ω matrix 
      updateΩ!(vcm)
      updateΩest!(vcm)
      updateβ!(vcm)
  
      # output
      if niters == 0
        niters = maxiters
      end
   
      # 
      if verbose 
        return vcm, obj, niters, objvec
      else
        return vcm, obj, niters, zeros(0)
      end 

end 

"""
    vcselect(Y, V; penfun=NoPenalty(), λ=1.0, penwt=[ones(length(V)-1); 0.0],
                standardize=true, maxiters=1000, tol=1e-6, verbose=false, checktype=true)

"""
<<<<<<< HEAD
function vcselectpath(
    y       :: AbstractVector{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nlambda :: Int = 100, 
    λpath   :: AbstractVector{T} = T[],
    σ2      :: AbstractVector{T} = ones(T, length(V)),
    maxiter :: Int = 1000,
    tol     :: AbstractFloat = 1e-8,
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

        # initialize arrays  
        σ2path = zeros(T, length(V), nlambda)
        objpath = zeros(T, nlambda)
        niterspath = zeros(Int, nlambda)

        # create solution path 
        for iter in 1:nlambda 
            σ2, objpath[iter], niterspath[iter], = 
                    vcselect(y, V; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                    σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)
            σ2path[:, iter] = σ2 
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, niterspath, = vcselect(y, V; 
                penfun=penfun, maxiter=maxiter, tol=tol, verbose=verbose)
    end 

    # output 
    return σ2path, objpath, λpath, niterspath
=======
function vcselect(
    Y           :: AbstractVecOrMat{T},
    V           :: AbstractVector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(length(V)-1); 0.0],
    standardize :: Bool = true, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-6,
    verbose     :: Bool = false,
    checktype   :: Bool = true 
    ) where {T <: Real}

    vcmtmp = VCModel(Y, V)
    _, obj, niters, objvec = vcselect!(vcmtmp; penfun=penfun, λ=λ, penwt=penwt, 
            standardize=standardize, maxiters=maxiters, tol=tol, verbose=verbose, 
            checktype=checktype)

    return vcmtmp.Σ, obj, niters, objvec
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

end 