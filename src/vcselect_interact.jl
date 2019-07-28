"""
    vcselect(y, V1, V2; penfun, λ, penwt, σ2, maxiter, tol, verbose)

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

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) && !iszero(λ) && !iszero(penwt[j])
                if isinf(penwt[j])
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
            niters = iter
            break
        end

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
    vcselect(y, G, trt; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where 
`Ω = (σ2[1]*V[1] + σ2int[1]*Vint[1]) + ... (σ2[m]*V[m] + σ2int[m]*Vint[m]) + σ2[end]*V[end]`, 
`m` is no. of groups, and `V[i]` and `Vint[i]` are i-th covariance matrices for main 
effect and interaction effect, respectively. Minimization is achieved via majorization
minimization (MM) algorithm. `V[i]` and `Vint[i]` are either included or excluded together.

# Input
- `y`: nx1 trait/response vector
- `G`: mx1 vector of genotype matrices (matrix of minor allele counts) for each SNP-set, 
    (G[1],G[2],...,G[m]) where all G[i] have n rows 
- `trt`: nx1 vector or nxn diagonal matrix whose entries indicate treatment status 
    of each individual 

# Keyword
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty()
- `λ`: penalty strength, default is 1.0
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `σ2`: (m+1)x1 initial values for genetic effect, default is (1,1,...,1), 
    note: m+1 because it includes residual error variance component 
- `σ2int`: mx1 initial values for interaction effect, default is (1,1,...,1)
- `B`: projection matrix 
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
    y        :: AbstractVector{T},
    G        :: Vector{Matrix{T}},
    trt      :: AbstractVecOrMat{S};
    penfun   :: Penalty = NoPenalty(),
    λ        :: T = one(T),
    penwt    :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    σ2       :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int    :: AbstractVector{T} = ones(T, length(G)),
    B        :: AbstractMatrix{T} = zeros(T, length(y), 0),
    Ω        :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
    Ωinv     :: AbstractMatrix{T} = zeros(T, length(y), length(y)),
    maxiter  :: Int = 1000,
    tol      :: AbstractFloat = 1e-5,
    standardize :: Bool = true, 
    verbose  :: Bool = false
    ) where {T, S <: Real} 

    # handle errors 
    #@assert length(y) == size(trt, 1) "trait vector `y` and treatment status vector `T` must have the same length!\n"
    @assert length(G) + 1 == length(σ2) "length of initial estimates `σ2` must be a number of genes + 1 \n"
    @assert length(G) == length(σ2int) "vector of genotype matrices `G` and initial estimates `σ2int` must have the same length!\n"
    @assert penfun ∈ [L1Penalty(), NoPenalty()]  "available penalty functions are NoPenalty() and L1Penalty()!\n"

    # assign  
    n = length(y)       # no. observations
    m = length(G)       # no. genes
    ϵ = convert(T, 1e-8)

    # construct `V` and `Vint` 
    V = Vector{Matrix{Float64}}(undef, m + 1)
    Vint = Vector{Matrix{Float64}}(undef, m)
    if size(trt, 2) == 1 
        trtmat = Diagonal(trt)
    else 
        trtmat = trt 
    end 
    for i in 1:m
        V[i] = G[i] * G[i]'
        Vint[i] = trtmat * V[i] * trtmat' 
        if !isempty(B)
            V[i] = BLAS.gemm('T', 'N', B, V[i] * B)
            Vint[i] = BLAS.gemm('T', 'N', B, Vint[i] * B)
        end 
        # divide by frobenius norm if standardize=true
        if standardize 
            V[i] ./= norm(V[i])
            Vint[i] ./= norm(Vint[i])
        end 
    end 
    # divide by frobenius norm if standardize=true
    if standardize
        V[end] = Matrix(I, n, n) ./ √n
    else 
        V[end] = Matrix(I, n, n)
    end 

    if verbose 
        σ2, σ2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; penfun=penfun, λ=λ, 
                penwt=penwt, σ2=σ2, σ2int=σ2int, Ω=Ω, Ωinv=Ωinv, maxiter=maxiter, tol=tol, verbose=true)
        return σ2, σ2int, obj, niters, Ω, objvec;
    else 
        σ2, σ2int, obj, niters, Ω = vcselect(y, V, Vint; penfun=penfun, λ=λ, 
                penwt=penwt, σ2=σ2, σ2int=σ2int, Ω=Ω, Ωinv=Ωinv, maxiter=maxiter, tol=tol)
        return σ2, σ2int, obj, niters, Ω;
    end

end 

"""

"""
function vcselect( 
    y           :: AbstractVector{T},
    X           :: AbstractVecOrMat{T},
    G           :: Vector{Matrix{T}},
    trt         :: AbstractVecOrMat{S};
    penfun      :: Penalty = NoPenalty(),
    λ           :: T = one(T), 
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    Ω           :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5,
    standardize :: Bool = true, 
    verbose     :: Bool = false
    ) where {T, S <: Real}

    # assign constants 
    m = length(G) # no. genes
    n = length(y)

    # project onto nullspace 
    ynew, _, B = nullprojection(y, X, G; covariance=false) 
    if size(trt, 2) == 1
        trtmat = Diagonal(trt)                                    
    else 
        trtmat = trt 
    end 

    # call vcselect 
    if verbose 
        σ2, σ2int, obj, niters, _, objvec = vcselect(ynew, G, trtmat; penfun=penfun, λ=λ, 
            penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
            standardize=standardize, verbose=true)
    else 
        σ2, σ2int, obj, niters, = vcselect(ynew, G, trtmat; penfun=penfun, λ=λ, 
            penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
            standardize=standardize)
    end 
    
    # create V and Vint 
    V = Vector{Matrix{Float64}}(undef, m + 1)
    Vint = Vector{Matrix{Float64}}(undef, m)
    for i in 1:m
        V[i] = G[i] * G[i]'
        Vint[i] = trtmat * V[i] * trtmat'
        if standardize 
            V[i] ./= norm(V[i])
            Vint[i] ./= norm(Vint[i])
        end 
    end 
    if standardize
        V[end] = Matrix(I, n, n) ./ √n
    else 
        V[end] = Matrix(I, n, n)
    end 

    # # construct overall covariance matrix 
    # Ω = fill!(Ω, 0)     # covariance matrix 
    # for i in 1:m
    #     if iszero(σ2[i]) && iszero(σ2int[i])
    #         continue 
    #     else 
    #         axpy!(σ2[i], V[i], Ω) 
    #         axpy!(σ2int[i], Vint[i], Ω) 
    #     end 
    # end
    # axpy!(σ2[end], V[end], Ω)

    # estimate fixed effects 
    #β = betaestimate(y, X, Ω)
    β = betaestimate(y, X, V, Vint, σ2, σ2int)

    # output 
    if verbose
        σ2, σ2int, β, obj, niters, Ω, objvec
    else 
        σ2, σ2int, β, obj, niters, Ω
    end 

end 

"""
    vcselectpath()

# Input 
- `y`: response vector
- `X`: design matrix (if exists)
- `G`: vector of genotype matrices, i.e. `(G[1], ..., G[m])`, where `m` is no. of genes
- `trt`: vector or diagonal matrix of treatment status.

# Keyword arguments
-`penfun`: penalty function (e.g. `NoPenalty()`, `L1Penalty()`), default is `NoPenalty()`
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
- `standardize`: logical flag for covariance matrix standardization, default is `true`.
    If true, `V` and `Vint` are standardized by its Frobenius norm

# Output 
- `σ̂2path`: matrix of estimated variance components for genetic main effects 
- `σ̂2intpath`: matrix of estimated variance components for interaction effects 
- `β̂path`: matrix of fixed effects parameter estimates 
* `objpath`: vector of objective value at `σ̂2` and `σ̂2int`
* `λpath`: the actual sequence of `λ` values used
* `niterspath`: vector of the number of iterations to convergence.

"""
function vcselectpath(
    y           :: AbstractVector{T},
    G           :: Vector{Matrix{T}},
    trt         :: AbstractVecOrMat{S};
    penfun      :: Penalty = NoPenalty(),
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    nλ          :: Int = 100, 
    λpath       :: AbstractVector{T} = T[], 
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    B           :: AbstractMatrix{T} = zeros(T, length(y), 0),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5,
    standardize :: Bool = true
    ) where {T, S <: Real}

    if penfun != NoPenalty()
        # assign 
        m = length(G) 

        # create a lambda grid if not specified
        if isempty(λpath)
            maxλ, V, Vint = maxlambda(y, G, trt; penfun=penfun, penwt=penwt, 
                    maxiter=maxiter, tol=tol, standardize=standardize, B=B)
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
                vcselect(y, G, trt; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                σ2=σ2, σ2int=σ2int, B=B, maxiter=maxiter, tol=tol, standardize=standardize)
            σ2path[:, iter] = σ2
            σ2intpath[:, iter] = σ2int
        end 

    else # if no penalty, no lambda grid 
        σ2path, σ2intpath, objpath, niterspath, = vcselect(y, G, trt; penfun=penfun, 
            σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, standardize=standardize)

    end 

    # output 
    return σ2path, σ2intpath, objpath, λpath, niterspath

end 

"""

"""
function vcselectpath(
    y           :: AbstractVector{T},
    X           :: AbstractVecOrMat{T},
    G           :: Vector{Matrix{T}},
    trt         :: AbstractVecOrMat{S};
    penfun      :: Penalty = NoPenalty(),
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    nλ          :: Int = 100, 
    λpath       :: AbstractVector{T} = T[], 
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-5,
    standardize :: Bool = true
    ) where {T, S <: Real}

    # assign constants 
    m, n = length(G), length(y)

    # project y onto nullspace of X 
    ynew, _, B = nullprojection(y, X, G; covariance=false)

    #
    σ2path, σ2intpath, objpath, λpath, niterspath = vcselectpath(ynew, G, trt; 
            penfun=penfun, penwt=penwt, nλ=nλ, λpath=λpath, σ2=σ2, σ2int=σ2int,
            B=B, maxiter=maxiter, tol=tol, standardize=standardize)

    # create V and Vint 
    V = Vector{Matrix{Float64}}(undef, m + 1)
    Vint = Vector{Matrix{Float64}}(undef, m)
    if size(trt, 2) == 1
        trtmat = Diagonal(trt)                                    
    else 
        trtmat = trt 
    end 
    for i in 1:m
        V[i] = G[i] * G[i]'
        Vint[i] = trtmat * V[i] * trtmat'
        if standardize 
            V[i] ./= norm(V[i])
            Vint[i] ./= norm(Vint[i])
        end 
    end 
    if standardize
        V[end] = Matrix(I, n, n) ./ √n
    else 
        V[end] = Matrix(I, n, n)
    end 

    # obtain estimates for fixed effects parameter 
    βpath = zeros(T, size(X, 2), length(λpath))
    for iter in 1:length(λpath)
        βpath[:, iter] .= betaestimate(y, X, V, Vint, view(σ2path, :, iter), 
                view(σ2intpath, :, iter))
    end 



    # output 
    return σ2path, σ2intpath, βpath, objpath, λpath, niterspath
end 