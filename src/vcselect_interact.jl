"""
<<<<<<< HEAD
    vcselect(y, V, Vint; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]` and `V[end] = I`
Minimization is achieved via majorization-minimization (MM) algorithm. 
`V1[i]` and `V2[i]` are either included or excluded together.

# Input
- `y`: response vector
- `V`: vector of covariance matrices for main effect, (V[1],V[2],...,V[m],I)
- `Vint`: vector of covariance matrices, (Vint[1],Vint[2],...,Vint[m])
    note that each `V` has length m+1 while `Vint` has length m; 
    V[end] should be identity matrix or identity matrix divided by √n if standardized 

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
- `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
    Default is true 

# Output
- `σ2`: vector of estimated variance components 
- `obj`: objective value at the estimated variance components 
- `niters`: number of iterations to convergence
- `Ω`: covariance matrix evaluated at the estimated variance components
- `objvec`: vector of objective values at each iteration 
"""
function vcselect( 
    y           :: AbstractVector{T},
    V           :: Vector{Matrix{T}},
    Vint        :: Vector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    λ           :: T = one(T), 
    penwt       :: AbstractVector{T} = [ones(T, length(Vint)); zero(T)],
    σ2          :: AbstractVector{T} = ones(T, length(V)),
    σ2int       :: AbstractVector{T} = ones(T, length(Vint)),
    Ω           :: AbstractMatrix{T} = zeros(T, size(V[1])), 
    Ωinv        :: AbstractMatrix{T} = zeros(T, size(V[1])),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5,
    verbose     :: Bool = false
    ) where {T <: Real} 

    # # handle errors 
    # @assert length(V) == length(σ2) "V and σ2 must have the same length!\n"
    # @assert length(Vint) == length(σ2int) "Vint and σ2int must have the same length!\n"
    # @assert penfun ∈ [L1Penalty(), NoPenalty()]  "available penalty functions are NoPenalty() and L1Penalty()!\n"

    # assign 
    n = length(y)       # no. observations
    m = length(Vint)    # no. groups
    ϵ = convert(T, 1e-8)

    # construct overall covariance matrix 
    Ω = fill!(Ω, 0)     # covariance matrix 
    for j in 1:m
        axpy!(σ2[j], V[j], Ω) 
        axpy!(σ2int[j], Vint[j], Ω) 
    end
    Ω .+= σ2[end] .* V[end]

    # inverse of Ω
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv .= inv(Ωchol) 
    v = Ωinv * y
    w = similar(v) 

    # objective value 
    loglConst = (1//2) * n * log(2π) 
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)  
    pen = 0.0
    for j in 1:m
        if iszero(σ2[j]) && iszero(σ2int[j])
            continue 
        else 
            pen += penwt[j] * value(penfun, √(σ2[j] + σ2int[j]))
        end 
    end
    obj += loglConst + λ * pen

    # display 
    if verbose
        println("iter = 0")
        println("σ2, σ2int = $(σ2), $(σ2int)")
        println("obj  = ", obj)
        objvec = obj
    end     

    # MM loop 
    niters = 0 
    for iter in 1:maxiter 
        # update variance components
        fill!(Ω, 0)
        for j in 1:m

            # move onto the next variance component if previous iterate is 0
            if iszero(σ2[j]) && iszero(σ2int[j])
                continue 
            end 

            # update σ2
            const1 = dot(Ωinv, V[j]) # const1 = tr(Ωinv * V[j])
            mul!(w, V[j], v)
            const2 = dot(w, v)       # const2 = y' * Ωinv * V[j] * Ωinv * y
        
            # update σ2int
            const1int = dot(Ωinv, Vint[j]) # const1 = tr(Ωinv * Vint[j])
            mul!(w, Vint[j], v)
            const2int = dot(w, v)          # const2 = y' * Ωinv * Vint[j] * Ωinv * y
=======
    vcselectpath!(vcm; penfun, penwt, nλ, λpath, maxiters, tol, verbose)

Generate solution path of variance components along varying lambda values.

# Input
- `vcm`: VCintModel

# Keyword 
- `penfun`: penalty function (e.g. `NoPenalty()`, `L1Penalty()`, `MCPPenalty()`), 
            default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nλ`: number of tuning parameter values, default is 100
- `λpath`: a user supplied `λ` sequence. Typically the program computes its own `λ` 
        sequence based on `nλ`; supplying `λpath` overrides this
- `maxiter`: maximum number of iteration for MM loop, default is 1000
- `standardize`: logical flag for covariance matrix standardization, default is `false`.
    If true, `V[i]` and `Vint[i]` are standardized by its Frobenius norm
- `tol`: convergence tolerance, default is `1e-5`
- `verbose`: display switch, default is false 

# Output 
- `Σ̂path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `Σ̂intpath`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `β̂path`: matrix of fixed parameter estimates at each `λ`; i-th column is an estimate 
        at `λpath[i]`
- `λpath`: sequence of `λ` values used
- `objpath`: vector of objective values at each `λ`; i-th value corresponds to `λpath[i]`
- `niterspath`: vector of no. of iterations to convergence at each `λ`; i-th value 
        corresponds to `λpath[i]`
"""
function vcselectpath!(
    vcm          :: VCintModel; 
    penfun       :: Penalty = NoPenalty(), 
    penwt        :: AbstractArray = [ones(ngroups(vcm)); 0],
    nλ           :: Int = 100, 
    λpath        :: AbstractArray = zeros(0), 
    maxiters     :: Int = 1000, 
    standardize  :: Bool = false, 
    tol          :: AbstractFloat = 1e-5
    )

    # handle errors 
    @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty() or L1Penalty()!\n"
    @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
    @assert size(λpath, 2) <= 1 "λpath must be one-dimensional array!\n"
    @assert maxiters > 0 "maxiters should be a positive integer!\n"

    # type 
    T = eltype(vcm.Y)

    # generate solution path based on penalty
    if penfun != NoPenalty()

        # create a lambda grid if not specified 
        if isempty(λpath) 
            maxλ, = maxlambda(vcm.Y, vcm.V, vcm.Vint; penfun=penfun, penwt=penwt, 
                            standardize=standardize)
            λpath = range(0, stop=maxλ, length=nλ)
        else # if lambda grid specified, make sure nlambda matches 
            nλ = length(λpath)
        end 

        # initialize arrays 
        objpath = zeros(T, nλ)
        niterspath = zeros(Int, nλ)
        Σ̂path = zeros(T, ngroups(vcm) + 1, nλ)
        Σ̂intpath = zeros(T, ngroups(vcm), nλ)
        β̂path = zeros(T, ncovariates(vcm), nλ)

        # solution path 
        for iter in 1:nλ
            _, objpath[iter], niterspath[iter] = 
                    vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                    maxiters=maxiters, tol=tol, verbose=false, checktype=false,
                    standardize=standardize)
            Σ̂path[:, iter] .= vcm.Σ
            Σ̂intpath[:, iter] .= vcm.Σint
            β̂path[:, iter] .= vcm.β
        end

        return Σ̂path, Σ̂intpath, β̂path, λpath, objpath, niterspath

    else # if no penalty, there is no lambda grid 
        _, objpath, niterspath = vcselect!(vcm; penfun=penfun, penwt=penwt, 
            maxiters=maxiters, tol=tol, verbose=false, checktype=false,
            standardize=standardize)

        return vcm.Σ, vcm.Σint, vcm.β, zeros(1), objpath, niterspath 
    end 

end 

"""
    vcselect!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

# Input 
- `vcm`: VCintModel

# Keyword 
- `penfun`: penalty function (e.g. `NoPenalty()`, `L1Penalty()`, `MCPPenalty()`), 
            default is NoPenalty()
- `λ`: tuning parameter, default is 1.0   
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `maxiters`: maximum number of iterations, default is 1000
- `standardize`: logical flag for covariance matrix standardization, default is `false`.
    If true, `V[i]` and `Vint[i]` is standardized by its Frobenius norm, and parameter 
    estimates are returned on the original scale
- `tol`: convergence tolerance, default is `1e-5`
- `verbose`: display switch, default is false 
- `checktype`: check argument type switch, default is true
- `objvec`: vector of objvective values at each iteration 

# Output 
- `vcm`: VCintModel with updated `Σ`, `Σint` and `β` 
    Access estimates with `vcm.Σ`, `vcm.Σint` and `vcm.β`
- `obj`: objective value at convergence 
- `niters`: number of iterations to convergence 
"""
function vcselect!(
    vcm          :: VCintModel;
    penfun       :: Penalty = NoPenalty(),
    λ            :: Real = 1.0,
    penwt        :: AbstractVector = [ones(ngroups(vcm)); 0.0],
    standardize  :: Bool = false, 
    maxiters     :: Int = 1000,
    tol          :: Real = 1e-5,
    verbose      :: Bool = false,
    checktype    :: Bool = true 
    )

    # handle errors 
    if checktype 
        @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty() or L1Penalty()!\n"
        @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
        @assert maxiters > 0 "maxiters should be a positive integer!\n"
        # start point has to be strictly positive
        @assert all(vcm.Σ .> 0) "starting Σ should be strictly positive or non-zero matrices"
    end 

    # update weight with reciprocal of frobenius norm 
    if standardize 
        vcm.wt .= 1 ./ norm.(vcm.V)
        vcm.wt_int .= 1 ./ norm.(vcm.Vint)
    end 

    # update with mm algorithm 
    _, obj, niters, objvec = mm_update_σ2!(vcm; penfun=penfun, λ=λ, penwt=penwt, 
            maxiters=maxiters, tol=tol, verbose=verbose)

    # output 
    return vcm, obj, niters, objvec

end 

"""
    mm_update_σ2!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `σ2` using MM algorithm. 
"""
function mm_update_σ2!(
    vcm         :: VCintModel;
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize :: Bool = false, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-5,
    verbose     :: Bool = false 
    )

    # initialize algorithm 
    n = size(vcm)[1]
    m = ngroups(vcm) 

    # initial objective value 
    updateΩ!(vcm)
    update_arrays!(vcm)
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)

    # display 
    if verbose
        println("iter  = 0")
        println("σ2    = ", vcm.Σ)
        println("σ2int = ", vcm.Σint)
        println("obj   = ", obj)
        objvec = obj 
    end    

    σ2tmp = zeros(m + 1)
    σ2inttmp = zeros(m)

    # MM loop 
    niters = 0
    for iter in 1:maxiters
        # update variance components
        for j in 1:m
            # move onto the next variance component if previous iterate is 0
            if iszero(vcm.Σ[j]) && iszero(vcm.Σint[j])
                σ2tmp[j] = 0 
                σ2inttmp[j] = 0
                continue 
            end 

            # compute constants for Σ
            const1 = dot(vcm.Ωinv, vcm.V[j]) # const1 = tr(Ωinv * V[j])
            mul!(vcm.Mnxd, vcm.V[j], vcm.ΩinvY)
            const2 = dot(vcm.Mnxd, vcm.ΩinvY) # const2 = y' * Ωinv * V[j] * Ωinv * y

            # compute constants for Σint 
            const1int = dot(vcm.Ωinv, vcm.Vint[j]) # const1 = tr(Ωinv * Vint[j])
            mul!(vcm.Mnxd, vcm.Vint[j], vcm.ΩinvY)
            const2int = dot(vcm.Mnxd, vcm.ΩinvY) # const2 = y' * Ωinv * Vint[j] * Ωinv * y
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && !iszero(λ) && !iszero(penwt[j])
                if isinf(penwt[j])
<<<<<<< HEAD
                    σ2[j] = zero(T)
                    σ2int[j] = zero(T)
                    continue 
                else
                    pen = λ * penwt[j] / sqrt(σ2[j] + σ2int[j])
                    # L1 penalty 
                    if isa(penfun, L1Penalty)  
                        σ2[j] = σ2[j] * 
                            √(const2 / (const1 + pen))
                        σ2int[j] = σ2int[j] * 
                            √(const2int / (const1int + pen))
                    # MCP penalty 
                    elseif isa(penfun, MCPPenalty) 
                        if σ2[j] <= (penfun.γ * λ)^2
                            σ2[j] = σ2[j] * 
                                √(const2 / (const1 + pen - (1 / penfun.γ)))
                        else
                            σ2[j] = σ2[j] * 
                                √(const2 / const1)
                        end 
                        if σ2int[j] <= (penfun.γ * λ)^2
                            σ2int[j] = σ2int[j] * 
                                √(const2int / (const1int + pen - (1 / penfun.γ)))
                        else
                            σ2int[j] = σ2int[j] * 
                                √(const2int / const1int)
                        end  
                    end 
                end 
                
            # update under no penalty 
            else
                σ2[j] = σ2[j] * √(const2 / const1)
                σ2int[j] = σ2int[j] * √(const2int / const1int)
            end 

            # update overall covariance matrix 
            axpy!(σ2[j], V[j], Ω) 
            axpy!(σ2int[j], Vint[j], Ω) 

        end # end of for loop over j  

        # update last variance component 
        σ2[end] = σ2[end] * √(dot(v, v) / tr(Ωinv))
        σ2[end] = clamp(σ2[end], ϵ, T(Inf))

        # update overall covariance matrix 
        axpy!(σ2[end], V[end], Ω) 

        # update Ωchol, Ωinv, v 
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(v, Ωinv, y)

        # update objective value 
        objold = obj
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)
        pen = 0.0
        for j in 1:m
            if iszero(σ2[j]) && iszero(σ2int[j])
                continue 
            else 
                pen += penwt[j] * value(penfun, √(σ2[j] + σ2int[j]))
            end
        end
        obj += loglConst + λ * pen
    
        # display 
        if verbose
            println("iter = ", iter)
            println("σ2, σ2int = $(σ2), $(σ2int)")
            println("obj  = ", obj)
            objvec = [objvec; obj]
        end    

        # check convergence
        if abs(obj - objold) < tol * (abs(objold) + 1)
=======
                    σ2tmp[j] = 0
                    σ2inttmp[j] = 0
                    continue 
                else 
                    pen = λ * penwt[j] / sqrt(vcm.Σ[j] + vcm.Σint[j])
                    # L1 penalty 
                    if isa(penfun, L1Penalty)  
                        σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + (pen / vcm.wt[j])))
                        σ2inttmp[j] = vcm.Σint[j] * 
                                √(const2int / (const1int + (pen / vcm.wt_int[j])))
                    # MCP penalty 
                    elseif isa(penfun, MCPPenalty)
                        if σ2tmp[j] <= (penfun.γ * λ)^2
                            σ2tmp[j] = vcm.Σ[j] * 
                                √(const2 / (const1 + pen - (1 / penfun.γ)))
                        else
                            σ2tmp[j] = vcm.Σ[j] * 
                                √(const2 / const1)
                        end 
                        if σ2inttmp[j] <= (penfun.γ * λ)^2
                            σ2inttmp[j] = vcm.Σint[j] * 
                                √(const2int / (const1int + pen - (1 / penfun.γ)))
                        else
                            σ2inttmp[j] = vcm.Σint[j] * 
                                √(const2int / const1int)
                        end  

                    end 
                end  
            # update under no penalty 
            else
                σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
                σ2inttmp[j] = vcm.Σint[j] * √(const2int / const1int)  
            end

        end # end of for loop over j

        # update last variance component and Ω
        σ2tmp[end] = vcm.Σ[end] *  √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
        σ2tmp[end] = clamp(σ2tmp[end], tol, Inf)

        vcm.Σ .= σ2tmp
        vcm.Σint .= σ2inttmp

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
            println("σ2int  = ", vcm.Σint)
            println("obj  = ", obj)
            objvec = [objvec; obj] 
        end

        # check convergence
        if abs(obj - objold) < tol * (abs(obj) + 1)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
            niters = iter
            break
        end

<<<<<<< HEAD
    end # end of MM loop 

    # construct final Ω matrix 
    fill!(Ω, 0)
    for i in 1:m
        if iszero(σ2[i]) && iszero(σ2int[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
            axpy!(σ2int[i], Vint[i], Ω) # Ω .+= σ2int[i] * Vint[i]
        end 
    end 
    axpy!(σ2[end], V[end], Ω)

    # output
    if niters == 0
    niters = maxiter
    end

    if verbose 
        return σ2, σ2int, obj, niters, Ω, objvec;
    else 
        return σ2, σ2int, obj, niters, Ω;
    end

end 

"""

"""
function vcselect( 
    y           :: AbstractVector{T},
    X           :: AbstractVecOrMat{S},
    V           :: Vector{Matrix{T}},
    Vint        :: Vector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    λ           :: T = one(T), 
    penwt       :: AbstractVector{T} = [ones(T, length(Vint)); zero(T)],
    σ2          :: AbstractVector{T} = ones(T, length(V)),
    σ2int       :: AbstractVector{T} = ones(T, length(Vint)),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5,
    verbose     :: Bool = false
    ) where {T, S <: Real} 

    # project onto nullspace 
    ynew, Vnew, Vintnew, = nullprojection(y, X, V, Vint)

    # 
    if verbose 
        σ2, σ2int, obj, niters, Ω, objvec = vcselect(ynew, Vnew, Vintnew; penfun=penfun,
                λ=λ, penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter,
                tol=tol, verbose=true)
    else
        σ2, σ2int, obj, niters, Ω = vcselect(ynew, Vnew, Vintnew; penfun=penfun,
                λ=λ, penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter,
                tol=tol)
    end 

    β = betaestimate(y, X, V, Vint, σ2, σ2int)

    if verbose 
        return σ2, σ2int, β, obj, niters, Ω, objvec;
    else 
        return σ2, σ2int, β, obj, niters, Ω;
    end

end


"""

"""
function vcselectpath(
    y           :: AbstractVector{T},
    V           :: Vector{Matrix{T}},
    Vint        :: Vector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    penwt       :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nλ          :: Int = 100, 
    λpath       :: AbstractVector{T} = T[], 
    σ2          :: AbstractVector{T} = ones(T, length(V)),
    σ2int       :: AbstractVector{T} = ones(T, length(Vint)),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5
    ) where {T <: Real}

    if penfun != NoPenalty()
        # assign 
        m = length(Vint) 

        # create a lambda grid if not specified
        if isempty(λpath)
            maxλ = maxlambda(y, V, Vint; penfun=penfun, penwt=penwt, 
                    maxiter=maxiter, tol=tol)
            λpath = range(0, stop=maxλ, length=nλ)
        else 
            nλ = length(λpath)
        end 

        # initialize arrays 
        σ2path = Matrix{T}(undef, m+1, nλ) 
        σ2intpath = Matrix{T}(undef, m, nλ)  
        objpath = Vector{Float64}(undef, nλ) 
        niterspath = Vector{Int}(undef, nλ) 

        # create solution path 
        for iter in 1:nλ 
            σ2, σ2int, objpath[iter], niterspath[iter], = 
                vcselect(y, V, Vint; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol)
            σ2path[:, iter] = σ2
            σ2intpath[:, iter] = σ2int
        end 

    else # if no penalty, no lambda grid 
        σ2path, σ2intpath, objpath, niterspath, = vcselect(y, V, Vint; penfun=penfun, 
            σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol)

    end 

    # output 
    return σ2path, σ2intpath, λpath, objpath, niterspath

end 


"""
# Input 
- `y`: response vector
- `V`: vector of covariance matrices for main effect, (V[1],V[2],...,V[m],I)
- `Vint`: vector of covariance matrices, (Vint[1],Vint[2],...,Vint[m])
    note that each `V` has length m+1 while `Vint` has length m; 
    V[end] should be identity matrix or identity matrix divided by √n if standardized 

# Keyword arguments
- `penfun`: penalty function (e.g. `NoPenalty()`, `L1Penalty()`), default is `NoPenalty()`
- `penwt`: penalty weight, default is (1,1,...1,0). `penwt` is a vector of length m+1
- `nλ`: the number of lambda values, default is 100
- `λpath`: a user supplied lambda sequence. 
    Typically the program computes its own lambda sequence based on `nλ`; 
    supplying `λpath` overrides this
- `σ2`: initial estimates for main effects, default is (1,...,1)
    i-th element (i=1,...,m) of the vector indicates main effect for i-th gene 
    while the last element (i=m+1 or `σ2[end]`) is residual variance
- `σ2int`: initial estimates for interaction effects, default is (1,...,1) 
    i-th element of the vector indicates interaction effect for i-th gene 
- `maxiter`: maximum number of iterations, default is 1000
- `tol`: convergence tolerance, default is `1e-6`
- `fixedeffects`: logical flag indicating whether to return fixed effects estimates,
    an empty array is returned if `fixedeffects=false`. Default is FALSE

# Output 
- `σ̂2path`: matrix of estimated variance components for genetic main effects 
- `σ̂2intpath`: matrix of estimated variance components for interaction effects 
- `β̂path`: matrix of fixed effects parameter estimates 
- `objpath`: vector of objective value at `σ̂2` and `σ̂2int`
- `λpath`: the actual sequence of `λ` values used
- `niterspath`: vector of the number of iterations to convergence.
"""
function vcselectpath(
    y            :: AbstractVector{T},
    X            :: AbstractVecOrMat{S},
    V            :: Vector{Matrix{T}},
    Vint         :: Vector{Matrix{T}};
    penfun       :: Penalty = NoPenalty(),
    penwt        :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    nλ           :: Int = 100, 
    λpath        :: AbstractVector{T} = T[], 
    σ2           :: AbstractVector{T} = ones(T, length(V)),
    σ2int        :: AbstractVector{T} = ones(T, length(Vint)),
    maxiter      :: Int = 1000,
    tol          :: AbstractFloat = 1e-5,
    fixedeffects :: Bool = false 
    ) where {S, T <: Real}

    # project onto nullspace 
    ynew, Vnew, Vintnew, = nullprojection(y, X, V, Vint)

    # get solution path 
    σ2path, σ2intpath, λpath, objpath, niterspath = vcselectpath(ynew, Vnew, Vintnew; 
        penfun=penfun, penwt=penwt, nλ=nλ, λpath=λpath, σ2=σ2, σ2int=σ2int, 
        maxiter=maxiter, tol=tol)

    # get βpath if requested by user 
    if !fixedeffects 
        βpath = zeros(T, size(X, 2), 0)
    else 
        βpath = zeros(T, size(X, 2), length(λpath))
        for iter in 1:length(λpath)
            βpath[:, iter] .= betaestimate(y, X, V, Vint, view(σ2path, :, iter), 
                    view(σ2intpath, :, iter))
        end 
    end 

    # 
    return σ2path, σ2intpath, βpath, λpath, objpath, niterspath
    
end 

# """
#     vcselect(y, G, trt; penfun, λ, penwt, σ2, maxiter, tol, verbose)

# Select variance components at specified lambda by minimizing penalized negative 
# log-likelihood of variance component model. 
# The objective function to minimize is
#   `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
# where 
# `Ω = (σ2[1]*V[1] + σ2int[1]*Vint[1]) + ... (σ2[m]*V[m] + σ2int[m]*Vint[m]) + σ2[end]*V[end]`, 
# `m` is no. of groups, and `V[i]` and `Vint[i]` are i-th covariance matrices for main 
# effect and interaction effect, respectively. Minimization is achieved via majorization
# minimization (MM) algorithm. `V[i]` and `Vint[i]` are either included or excluded together.

# # Input
# - `y`: nx1 trait/response vector
# - `G`: mx1 vector of genotype matrices (matrix of minor allele counts) for each SNP-set, 
#     (G[1],G[2],...,G[m]) where all G[i] have n rows 
# - `trt`: nx1 vector or nxn diagonal matrix whose entries indicate treatment status 
#     of each individual 

# # Keyword
# - `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty()
# - `λ`: penalty strength, default is 1.0
# - `penwt`: vector of penalty weights, default is (1,1,...1,0)
# - `σ2`: (m+1)x1 initial values for genetic effect, default is (1,1,...,1), 
#     note: m+1 because it includes residual error variance component 
# - `σ2int`: mx1 initial values for interaction effect, default is (1,1,...,1)
# - `B`: projection matrix 
# - `Ω`: initial overall covariance matrix `Ω`
# - `Ωinv`: initial inverse matrix of overall covariance matrix `Ω`
# - `maxiter`: maximum number of iterations, default is 1000
# - `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
# - `verbose`: display switch, default is false 
# - `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
#     Default is true 

# # Output
# - `σ2`: vector of estimated variance components 
# - `obj`: objective value at the estimated variance components 
# - `niters`: number of iterations to convergence
# - `Ω`: covariance matrix evaluated at the estimated variance components
# - `objvec`: vector of objective values at each iteration 
# """
# function vcselect( 
#     y        :: AbstractVector{T},
#     G        :: Vector{Matrix{T}},
#     trt      :: AbstractVecOrMat{S};
#     penfun   :: Penalty = NoPenalty(),
#     λ        :: T = one(T),
#     penwt    :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
#     σ2       :: AbstractVector{T} = ones(T, length(G)+1),
#     σ2int    :: AbstractVector{T} = ones(T, length(G)),
#     B        :: AbstractMatrix{T} = zeros(T, length(y), 0),
#     Ω        :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
#     Ωinv     :: AbstractMatrix{T} = zeros(T, length(y), length(y)),
#     maxiter  :: Int = 1000,
#     tol      :: AbstractFloat = 1e-5,
#     standardize :: Bool = true, 
#     verbose  :: Bool = false
#     ) where {T, S <: Real} 

#     # handle errors 
#     #@assert length(y) == size(trt, 1) "trait vector `y` and treatment status vector `T` must have the same length!\n"
#     @assert length(G) + 1 == length(σ2) "length of initial estimates `σ2` must be a number of genes + 1 \n"
#     @assert length(G) == length(σ2int) "vector of genotype matrices `G` and initial estimates `σ2int` must have the same length!\n"
#     @assert penfun ∈ [L1Penalty(), NoPenalty()]  "available penalty functions are NoPenalty() and L1Penalty()!\n"

#     # assign  
#     n = length(y)       # no. observations
#     m = length(G)       # no. genes
#     ϵ = convert(T, 1e-8)

#     # construct `V` and `Vint` 
#     V = Vector{Matrix{Float64}}(undef, m + 1)
#     Vint = Vector{Matrix{Float64}}(undef, m)
#     if size(trt, 2) == 1 
#         trtmat = Diagonal(trt)
#     else 
#         trtmat = trt 
#     end 
#     for i in 1:m
#         V[i] = G[i] * G[i]'
#         Vint[i] = trtmat * V[i] * trtmat' 
#         if !isempty(B)
#             V[i] = BLAS.gemm('T', 'N', B, V[i] * B)
#             Vint[i] = BLAS.gemm('T', 'N', B, Vint[i] * B)
#         end 
#         # divide by frobenius norm if standardize=true
#         if standardize 
#             V[i] ./= norm(V[i])
#             Vint[i] ./= norm(Vint[i])
#         end 
#     end 
#     # divide by frobenius norm if standardize=true
#     if standardize
#         V[end] = Matrix(I, n, n) ./ √n
#     else 
#         V[end] = Matrix(I, n, n)
#     end 

#     if verbose 
#         σ2, σ2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; penfun=penfun, λ=λ, 
#                 penwt=penwt, σ2=σ2, σ2int=σ2int, Ω=Ω, Ωinv=Ωinv, maxiter=maxiter, tol=tol, verbose=true)
#         return σ2, σ2int, obj, niters, Ω, objvec;
#     else 
#         σ2, σ2int, obj, niters, Ω = vcselect(y, V, Vint; penfun=penfun, λ=λ, 
#                 penwt=penwt, σ2=σ2, σ2int=σ2int, Ω=Ω, Ωinv=Ωinv, maxiter=maxiter, tol=tol)
#         return σ2, σ2int, obj, niters, Ω;
#     end

# end 

# """
#     vcselect(y, X, G, trt; penfun, λ, penwt, σ2, σ2int, B, Ω, Ωinv, maxiter, tol, verbose)

# Select variance components at specified lambda by minimizing penalized negative 
# log-likelihood of variance component model. 
# The objective function to minimize is
#   `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
# where 
# `Ω = (σ2[1]*V[1] + σ2int[1]*Vint[1]) + ... (σ2[m]*V[m] + σ2int[m]*Vint[m]) + σ2[end]*V[end]`, 
# `m` is no. of groups, and `V[i]` and `Vint[i]` are i-th covariance matrices for main 
# effect and interaction effect, respectively. Minimization is achieved via majorization
# minimization (MM) algorithm. `V[i]` and `Vint[i]` are either included or excluded together.

# # Input
# - `y`: nx1 trait/response vector
# - `X`: covariate matrix 
# - `G`: mx1 vector of genotype matrices (matrix of minor allele counts) for each SNP-set, 
#     (G[1],G[2],...,G[m]) where all G[i] have n rows 
# - `trt`: nx1 vector or nxn diagonal matrix whose entries indicate treatment status 
#     of each individual 

# # Keyword
# - `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty()
# - `λ`: penalty strength, default is 1.0
# - `penwt`: vector of penalty weights, default is (1,1,...1,0)
# - `σ2`: (m+1)x1 initial values for genetic effect, default is (1,1,...,1), 
#     note: m+1 because it includes residual error variance component 
# - `σ2int`: mx1 initial values for interaction effect, default is (1,1,...,1)
# - `B`: projection matrix 
# - `Ω`: initial overall covariance matrix `Ω`
# - `Ωinv`: initial inverse matrix of overall covariance matrix `Ω`
# - `maxiter`: maximum number of iterations, default is 1000
# - `tol`: tolerance in difference of objective values for MM loop, default is 1e-6
# - `verbose`: display switch, default is false 
# - `checkfrobnorm`: if true, makes sures elements of `V` have frobenius norm 1.
#     Default is true 

# # Output
# - `σ2`: vector of estimated variance components 
# - `σ2int`: vector of estimated variance components for interaction matrices 
# - `β`: vector of estimated fixed effects coefficients 
# - `obj`: objective value at the estimated variance components 
# - `niters`: number of iterations to convergence
# - `Ω`: covariance matrix evaluated at the estimated variance components
# - `objvec`: vector of objective values at each iteration 
# """
# function vcselect( 
#     y           :: AbstractVector{T},
#     X           :: AbstractVecOrMat{T},
#     G           :: Vector{Matrix{T}},
#     trt         :: AbstractVecOrMat{S};
#     penfun      :: Penalty = NoPenalty(),
#     λ           :: T = one(T), 
#     penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
#     σ2          :: AbstractVector{T} = ones(T, length(G)+1),
#     σ2int       :: AbstractVector{T} = ones(T, length(G)),
#     Ω           :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
#     maxiter     :: Int = 1000,
#     tol         :: AbstractFloat = 1e-5,
#     standardize :: Bool = true, 
#     verbose     :: Bool = false
#     ) where {T, S <: Real}

#     # assign constants 
#     m = length(G) # no. genes
#     n = length(y)

#     # project onto nullspace 
#     ynew, _, B = nullprojection(y, X, G; covariance=false) 
#     if size(trt, 2) == 1
#         trtmat = Diagonal(trt)                                    
#     else 
#         trtmat = trt 
#     end 

#     # call vcselect 
#     if verbose 
#         σ2, σ2int, obj, niters, _, objvec = vcselect(ynew, G, trtmat; penfun=penfun, λ=λ, 
#             penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
#             standardize=standardize, verbose=true)
#     else 
#         σ2, σ2int, obj, niters, = vcselect(ynew, G, trtmat; penfun=penfun, λ=λ, 
#             penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
#             standardize=standardize)
#     end 
    
#     # create V and Vint 
#     V = Vector{Matrix{Float64}}(undef, m + 1)
#     Vint = Vector{Matrix{Float64}}(undef, m)
#     for i in 1:m
#         V[i] = G[i] * G[i]'
#         Vint[i] = trtmat * V[i] * trtmat'
#         if standardize 
#             V[i] ./= norm(V[i])
#             Vint[i] ./= norm(Vint[i])
#         end 
#     end 
#     if standardize
#         V[end] = Matrix(I, n, n) ./ √n
#     else 
#         V[end] = Matrix(I, n, n)
#     end 

#     β = betaestimate(y, X, V, Vint, σ2, σ2int)

#     # output 
#     if verbose
#         σ2, σ2int, β, obj, niters, Ω, objvec
#     else 
#         σ2, σ2int, β, obj, niters, Ω
#     end 

# end 

# """
#     vcselectpath(y, G, trt; penfun, penwt, nλ, λpath, σ2, σ2int, maxiter, tol, standardize)

# # Input 
# - `y`: response vector
# - `X`: design matrix (if exists)
# - `G`: vector of genotype matrices, i.e. `(G[1], ..., G[m])`, where `m` is no. of genes
# - `trt`: vector or diagonal matrix of treatment status.

# # Keyword arguments
# -`penfun`: penalty function (e.g. `NoPenalty()`, `L1Penalty()`), default is `NoPenalty()`
# - `penwt`: penalty weight, default is (1,1,...1,0). `penwt` is a vector of length m+1
# - `nλ`: the number of lambda values, default is 100
# - `λpath`: a user supplied lambda sequence. 
#     Typically the program computes its own lambda sequence based on `nλ`; 
#     supplying `λpath` overrides this
# - `σ2`: initial estimates for main effects, default is (1,...,1)
#     i-th element (i=1,...,m) of the vector indicates main effect for i-th gene 
#     while the last element (i=m+1 or `σ2[end]`) is residual variance
# - `σ2int`: initial estimates for interaction effects, default is (1,...,1) 
#     i-th element of the vector indicates interaction effect for i-th gene 
# - `maxiter`: maximum number of iterations, default is 1000
# - `tol`: convergence tolerance, default is `1e-6`
# - `standardize`: logical flag for covariance matrix standardization, default is `true`.
#     If true, `V` and `Vint` are standardized by its Frobenius norm

# # Output 
# - `σ̂2path`: matrix of estimated variance components for genetic main effects 
# - `σ̂2intpath`: matrix of estimated variance components for interaction effects 
# - `β̂path`: matrix of fixed effects parameter estimates 
# - `objpath`: vector of objective value at `σ̂2` and `σ̂2int`
# - `λpath`: the actual sequence of `λ` values used
# - `niterspath`: vector of the number of iterations to convergence.
# """
# function vcselectpath(
#     y           :: AbstractVector{T},
#     G           :: Vector{Matrix{T}},
#     trt         :: AbstractVecOrMat{S};
#     penfun      :: Penalty = NoPenalty(),
#     penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
#     nλ          :: Int = 100, 
#     λpath       :: AbstractVector{T} = T[], 
#     σ2          :: AbstractVector{T} = ones(T, length(G)+1),
#     σ2int       :: AbstractVector{T} = ones(T, length(G)),
#     B           :: AbstractMatrix{T} = zeros(T, length(y), 0),
#     maxiter     :: Int = 1000,
#     tol         :: AbstractFloat = 1e-5,
#     standardize :: Bool = true
#     ) where {T, S <: Real}

#     if penfun != NoPenalty()
#         # assign 
#         m = length(G) 

#         # create a lambda grid if not specified
#         if isempty(λpath)
#             maxλ, V, Vint = maxlambda(y, G, trt; penfun=penfun, penwt=penwt, 
#                     maxiter=maxiter, tol=tol, standardize=standardize, B=B)
#             λpath = range(0, stop=maxλ, length=nλ)
#         else 
#             nλ = length(λpath)
#         end 

#         # initialize arrays 
#         σ2path = Matrix{T}(undef, m+1, nλ) 
#         σ2intpath = Matrix{T}(undef, m, nλ)  
#         objpath = Vector{Float64}(undef, nλ) 
#         niterspath = Vector{Int}(undef, nλ) 

#         # create solution path 
#         for iter in 1:nλ 
#             σ2, σ2int, objpath[iter], niterspath[iter], = 
#                 vcselect(y, G, trt; penfun=penfun, λ=λpath[iter], penwt=penwt, 
#                 σ2=σ2, σ2int=σ2int, B=B, maxiter=maxiter, tol=tol, standardize=standardize)
#             σ2path[:, iter] = σ2
#             σ2intpath[:, iter] = σ2int
#         end 

#     else # if no penalty, no lambda grid 
#         σ2path, σ2intpath, objpath, niterspath, = vcselect(y, G, trt; penfun=penfun, 
#             σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, standardize=standardize)

#     end 

#     # output 
#     return σ2path, σ2intpath, objpath, λpath, niterspath

# end 

# """

# """
# function vcselectpath(
#     y           :: AbstractVector{T},
#     X           :: AbstractVecOrMat{T},
#     G           :: Vector{Matrix{T}},
#     trt         :: AbstractVecOrMat{S};
#     penfun      :: Penalty = NoPenalty(),
#     penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
#     nλ          :: Int = 100, 
#     λpath       :: AbstractVector{T} = T[], 
#     σ2          :: AbstractVector{T} = ones(T, length(G)+1),
#     σ2int       :: AbstractVector{T} = ones(T, length(G)),
#     maxiter     :: Int = 1000,
#     tol         :: AbstractFloat = 1e-5,
#     standardize :: Bool = true
#     ) where {T, S <: Real}

#     # assign constants 
#     m, n = length(G), length(y)

#     # project y onto nullspace of X 
#     ynew, _, B = nullprojection(y, X, G; covariance=false)

#     #
#     σ2path, σ2intpath, objpath, λpath, niterspath = vcselectpath(ynew, G, trt; 
#             penfun=penfun, penwt=penwt, nλ=nλ, λpath=λpath, σ2=σ2, σ2int=σ2int,
#             B=B, maxiter=maxiter, tol=tol, standardize=standardize)

#     # create V and Vint 
#     V = Vector{Matrix{Float64}}(undef, m + 1)
#     Vint = Vector{Matrix{Float64}}(undef, m)
#     if size(trt, 2) == 1
#         trtmat = Diagonal(trt)                                    
#     else 
#         trtmat = trt 
#     end 
#     for i in 1:m
#         V[i] = G[i] * G[i]'
#         Vint[i] = trtmat * V[i] * trtmat'
#         if standardize 
#             V[i] ./= norm(V[i])
#             Vint[i] ./= norm(Vint[i])
#         end 
#     end 
#     if standardize
#         V[end] = Matrix(I, n, n) ./ √n
#     else 
#         V[end] = Matrix(I, n, n)
#     end 

#     # obtain estimates for fixed effects parameter 
#     βpath = zeros(T, size(X, 2), length(λpath))
#     for iter in 1:length(λpath)
#         βpath[:, iter] .= betaestimate(y, X, V, Vint, view(σ2path, :, iter), 
#                 view(σ2intpath, :, iter))
#     end 

#     # output 
#     return σ2path, σ2intpath, βpath, objpath, λpath, niterspath
# end 

=======
    end # end of iteration 

    # back to original scale  
    if standardize 
        vcm.Σ .*= vcm.wt
        vcm.Σint .*= vcm.wt_int
        vcm.wt .= ones(m + 1)
        vcm.wt_int .= ones(m)
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
    vcselect(Y, V, Vint; penfun=NoPenalty(), λ=1.0, penwt=[ones(length(V)-1); 0.0],
            standardize=true, maxiters=1000, tol=1e-5, verbose=false, checktype=true)

"""
function vcselect(
    Y           :: AbstractVecOrMat{T},
    V           :: AbstractVector{Matrix{T}},
    Vint        :: AbstractVector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(length(V)-1); 0.0],
    standardize :: Bool = false, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-5,
    verbose     :: Bool = false,
    checktype   :: Bool = true 
    ) where {T <: Real}

    vcmtmp = VCintModel(Y, V, Vint)
    _, obj, niters, objvec = vcselect!(vcmtmp; penfun=penfun, λ=λ, penwt=penwt, 
            standardize=standardize, maxiters=maxiters, tol=tol, verbose=verbose, 
            checktype=checktype)

    return vcmtmp.Σ, vcmtmp.Σint, obj, niters, objvec

end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
