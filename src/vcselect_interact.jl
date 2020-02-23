"""
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

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && !iszero(λ) && !iszero(penwt[j])
                if isinf(penwt[j])
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
            niters = iter
            break
        end

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