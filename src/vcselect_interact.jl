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
- `V1`: vector of covariance matrices, (V1[1],V1[2],...,V1[m],I/√n)
    note 1) each V1[i] needs to have frobenius norm 1, i=1,...,m
         2) `V1` and `V2` must have the same length
         3) `V1[end]` should be identity matrix divided by √n, where n is sample size 
- `V2`: vector of covariance matrices, (V2[1],V2[2],...,V2[m],I/√n)
    note that each V2[i] needs to have frobenius norm 1, and 
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
    y             :: AbstractVector{T},
    V1            :: Vector{Matrix{T}},
    V2            :: Vector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    θ             :: T = one(T),
    p             :: Real = 2,  
    penwt         :: AbstractVector{T} = [ones(T, length(V1)-1); zero(T)],
    σ2_1          :: AbstractVector{T} = ones(T, length(V1)),
    σ2_2          :: AbstractVector{T} = ones(T, length(V1)),
    Ω             :: AbstractMatrix{T} = zeros(T, size(V1[1])), 
    Ωinv          :: AbstractMatrix{T} = zeros(T, size(V1[1])),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-6,
    verbose       :: Bool = false
    ) where {T <: Real} 

    # handle errors 
    @assert length(V1) == length(V2) "V1 and V2 must have the same length!\n"
    @assert length(V1) == length(σ2_1) "V1 and σ2_1 must have the same length!\n"
    @assert length(V2) == length(σ2_2) "V2 and σ2_2 must have the same length!\n"
    @assert penfun ∈ [L1Penalty(), NoPenalty()]  "available penalty functions are NoPenalty() and L1Penalty()!\n"

    # 
    ϵ = convert(T, 1e-8)

    # initialize algorithm
    n = length(y)       # no. observations
    m = length(V1) - 1   # no. variance components
    Ω = fill!(Ω, 0)     # covariance matrix 
    for j in 1:(m + 1)
        Ω .+= σ2_1[j] .* V1[j] 
        Ω .+= σ2_2[j] .* V2[j] 
    end
    Ωchol = cholesky!(Symmetric(Ω))
    Ωinv = inv(Ωchol) 
    v = Ωinv * y
    w = similar(v) 

    # objective value 
    loglConst = (1//2) * n * log(2π) 
    obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)  
    pen = 0.0
    for j in 1:m
        if iszero(σ2_1[j]) && iszero(σ2_2[j])
            continue 
        else 
            pen += penwt[j] * value(penfun, √(σ2_1[j] + σ2_2[j]))
        end 
    end
    obj += loglConst + λ * pen

    # display 
    if verbose
        println("iter = 0")
        println("σ2_1, σ2_2 = $(σ2_1), $(σ2_2)")
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
            if iszero(σ2_1[j]) && iszero(σ2_2[j])
                continue 
            # set to 0 and move onto the next variance component if penalty weight is 0
            elseif iszero(penwt[j])
                σ2_1[j] = zero(T)
                σ2_2[j] = zero(T)
                continue 
            end 
            # update σ2_1
            const11 = dot(Ωinv, V1[j]) # const1 = tr(Ωinv * V1[j])
            mul!(w, V1[j], v)
            const12 = dot(w, v)         # const2 = y' * Ωinv * V1[j] * Ωinv * y
        
            # update σ2_2
            const21 = dot(Ωinv, V2[j]) # const1 = tr(Ωinv * V2[j])
            mul!(w, V2[j], v)
            const22 = dot(w, v)         # const2 = y' * Ωinv * V2[j] * Ωinv * y

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) 
                if isinf(penwt[j])
                    σ2_1[j] = zero(T)
                    σ2_2[j] = zero(T)
                    continue 
                else
                    pen = λ * penwt[j] / sqrt(σ2_1[j] + σ2_1[j])
                    # L1 penalty 
                    if isa(penfun, L1Penalty)  
                        σ2_1[j] = σ2_1[j] * 
                            √(const12 / (const11 + pen))
                        σ2_2[j] = σ2_2[j] * 
                            √(const22 / (const21 + pen))
                    end
                end 

                
            # update under no penalty 
            else
                σ2_1[j] = σ2_1[j] * √(const12 / const11)
                σ2_2[j] = σ2_2[j] * √(const22 / const21)
            end 

            # truncation step 
            if norm([σ2_1[j], σ2_2[j]], p) < θ
                σ2_1[j] = zero(T)
                σ2_2[j] = zero(T)
                continue
            end 

            # update overall covariance matrix 
            axpy!(σ2_1[j], V1[j], Ω) 
            axpy!(σ2_2[j], V2[j], Ω) 

        end # end of for loop over j  

        # update last variance component 
        σ2_1[end] = σ2_1[end] * √(dot(v, v) / tr(Ωinv))
        σ2_1[end] = clamp(σ2_1[end], ϵ, T(Inf))
        σ2_2[end] = σ2_1[end]

        # update overall covariance matrix 
        axpy!(σ2_1[end], V1[end], Ω) 

        # update Ωchol, Ωinv, v 
        Ωchol = cholesky!(Symmetric(Ω))
        Ωinv[:] = inv(Ωchol)
        mul!(v, Ωinv, y)

        # update objective value 
        objold = obj
        obj = (1//2) * logdet(Ωchol) + (1//2) * dot(y, v)
        pen = 0.0
        for j in 1:m
            if iszero(σ2_1[j]) && iszero(σ2_2[j])
                continue 
            else 
                pen += penwt[j] * value(penfun, √(σ2_1[j] + σ2_2[j]))
            end
        end
        obj += loglConst + λ * pen
    
        # display 
        if verbose
            println("iter = ", iter)
            println("σ2_1, σ2_2 = $(σ2_1), $(σ2_2)")
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
    for i in 1:(m + 1)
        if iszero(σ2_1[i]) && iszero(σ2_2[i])
            continue 
        else 
            axpy!(σ2_1[i], V1[i], Ω) # Ω .+= σ2[i] * V[i]
            axpy!(σ2_2[i], V2[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    # output
    if niters == 0
    niters = maxiter
    end

    if verbose 
        return σ2_1, σ2_2, obj, niters, Ω, objvec;
    else 
        return σ2_1, σ2_2, obj, niters, Ω;
    end


end 


"""

"""
function vcselect( 
    y             :: AbstractVector{T},
    X             :: AbstractVecOrMat{T},
    V1            :: Vector{Matrix{T}},
    V2            :: Vector{Matrix{T}};
    penfun        :: Penalty = NoPenalty(),
    λ             :: T = one(T),
    θ             :: T = one(T),
    p             :: Real = 2,  
    penwt         :: AbstractVector{T} = [ones(T, length(V1)-1); zero(T)],
    σ2_1          :: AbstractVector{T} = ones(T, length(V1)),
    σ2_2          :: AbstractVector{T} = ones(T, length(V1)),
    maxiter       :: Int = 1000,
    tol           :: AbstractFloat = 1e-6,
    verbose       :: Bool = false
    ) where {T <: Real} 

    # project onto nullspace 
    nvarcomps = length(V1)
    ynew, V1new, V2new = nullprojection(y, X, V1, V2)

    # call vcselect 
    if verbose 
        σ2_1, σ2_2, obj, niters, Ω, objvec = vcselect(ynew, V1new, V2new; penfun=penfun, λ=λ, θ=θ, p=p, penwt=penwt, 
            σ2_1=σ2_1, σ2_2=σ2_2, maxiter=maxiter, tol=tol, verbose=verbose)
    else 
        σ2_1, σ2_2, obj, niters, Ω = vcselect(ynew, V1new, V2new; penfun=penfun, λ=λ, θ=θ, p=p, penwt=penwt, 
            σ2_1=σ2_1, σ2_2=σ2_2, maxiter=maxiter, tol=tol, verbose=verbose)
    end 

    # update Ω with estimated variance components
    Ω = zeros(T, size(V1[1]))
    for i in 1:nvarcomps
        if iszero(σ2_1[i]) && iszero(σ2_2[i])
            continue 
        else 
            axpy!(σ2_1[i], V1[i], Ω) # Ω .+= σ2[i] * V[i]
            axpy!(σ2_2[i], V2[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    # estimate fixed effects 
    β = betaestimate(y, X, Ω)

    # output 
    if verbose 
        return σ2_1, σ2_2, β, obj, niters, Ω, objvec;
    else 
        return σ2_1, σ2_2, β, obj, niters, Ω;
    end 

end 

"""
    vcselect(y, G, trt; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = (σ2[1]*V[1] + σ2int[1]*Vint[1])... + σ2[end]*V[end]` and `V[end] = I`
Minimization is achieved via majorization-minimization (MM) algorithm. 
`V1[i]` and `V2[i]` are either included or excluded together.

# Input
- `y`: nx1 trait/response vector
- `G`: mx1 vector of genotype matrices (matrix of minor allele counts) for each SNP-set, (G[1],G[2],...,G[m])
- `T`: nx1 vector of treatment status for each individual 

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
    θ        :: T = one(T),
    p        :: Real = 2,  
    penwt    :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    σ2       :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int    :: AbstractVector{T} = ones(T, length(G)),
    B        :: AbstractMatrix{T} = zeros(T, length(y), 0),
    Ω        :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
    Ωinv     :: AbstractMatrix{T} = zeros(T, length(y), length(y)),
    maxiter  :: Int = 1000,
    tol      :: AbstractFloat = 1e-6,
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
            # set to 0 and move onto the next variance component if penalty weight is 0
            elseif iszero(penwt[j])
                σ2[j] = zero(T)
                σ2int[j] = zero(T)
                continue 
            end 
            # update σ2_1
            const1 = dot(Ωinv, V[j]) # const1 = tr(Ωinv * V1[j])
            mul!(w, V[j], v)
            const2 = dot(w, v)         # const2 = y' * Ωinv * V1[j] * Ωinv * y
        
            # update σ2_2
            const1int = dot(Ωinv, Vint[j]) # const1 = tr(Ωinv * V2[j])
            mul!(w, Vint[j], v)
            const2int = dot(w, v)         # const2 = y' * Ωinv * V2[j] * Ωinv * y

            # update variance component under specified penalty 
            if !isa(penfun, NoPenalty) 
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

            # truncation step 
            if norm([σ2[j], σ2int[j]], p) < θ
                σ2[j] = zero(T)
                σ2int[j] = zero(T)
                continue
            end 

            # update overall covariance matrix 
            axpy!(σ2[j], V[j], Ω) 
            axpy!(σ2int[j], Vint[j], Ω) 

        end # end of for loop over j  

        # update last variance component 
        σ2[end] = σ2[end] * √(dot(v, v) / tr(Ωinv))
        σ2[end] = clamp(σ2[end], ϵ, T(Inf))
        σ2int[end] = σ2int[end]

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

"""
function vcselect( 
    y           :: AbstractVector{T},
    X           :: AbstractVecOrMat{T},
    G           :: Vector{Matrix{T}},
    trt         :: AbstractVecOrMat{S};
    penfun      :: Penalty = NoPenalty(),
    λ           :: T = one(T),
    θ           :: T = one(T),
    p           :: Real = 2,  
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    Ω           :: AbstractMatrix{T} = zeros(T, length(y), length(y)), 
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-6,
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
            θ=θ, p=p, penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
            standardize=standardize, verbose=true)
    else 
        σ2, σ2int, obj, niters, = vcselect(ynew, G, trtmat; penfun=penfun, λ=λ, θ=θ,
            p=p, penwt=penwt, σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, B=B,
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

    # construct overall covariance matrix 
    Ω = fill!(Ω, 0)     # covariance matrix 
    for i in 1:m
        if iszero(σ2[i]) && iszero(σ2int[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) 
            axpy!(σ2int[i], Vint[i], Ω) 
        end 
    end
    axpy!(σ2[end], V[end], Ω)

    # estimate fixed effects 
    β = betaestimate(y, X, Ω)

    # output 
    if verbose 
        σ2, σ2int, β, obj, niters, Ω, objvec
    else 
        σ2, σ2int, β, obj, niters, Ω
    end 

end 

"""

"""
function vcselectpath(
    y           :: AbstractVector{T},
    G           :: Vector{Matrix{T}},
    trt         :: AbstractVecOrMat{S};
    penfun      :: Penalty = NoPenalty(),
    θ           :: T = one(T),
    p           :: Real = 2,  
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    nλ          :: Int = 100, 
    λpath       :: AbstractVector{T} = T[], 
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-6,
    standardize :: Bool = true
    ) where {T, S <: Real}

    if penfun != NoPenalty()
        # assign 
        m = length(G) 

        # create a lambda grid if not specified
        if isempty(λpath)

        else 
            nlambda = length(λpath)
        end 

        # initialize arrays 
        σ2path = Matrix{T}(undef, m+1, nlambda) 
        σ2intpath = Matrix{T}(undef, m, nlambda)  
        objpath = Vector{Float64}(undef, nlambda) 
        niterspath = Vector{Int}(undef, nlambda) 

        # create solution path 
        for iter in 1:nlambda 
            σ2, σ2int, objpath[iter], niterspath[iter], = 
                vcselect(y, G, trt; penfun=penfun, λ=λpath[iter], θ=θ, p=p, penwt=penwt, 
                σ2=σ2, σ2int=σ2int, maxiter=maxiter, tol=tol, standardize=standardize)
            σ2path[:, iter] = σ2
            σ2intpath[:, iter] = σ2int
        end 

    else # if no penalty, no lambda grid 
        σ2path, σ2intpath, objpath, niterspath, = vcselect(y, G, trt; penfun=penfun, θ=θ, p=p,
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
    θ           :: T = one(T),
    p           :: Real = 2,  
    penwt       :: AbstractVector{T} = [ones(T, length(G)); zero(T)],
    nλ          :: Int = 100, 
    λpath       :: AbstractVector{T} = T[], 
    σ2          :: AbstractVector{T} = ones(T, length(G)+1),
    σ2int       :: AbstractVector{T} = ones(T, length(G)),
    maxiter     :: Int = 1000,
    tol         :: AbstractFloat = 1e-6,
    standardize :: Bool = true, 
    verbose     :: Bool = false
    ) where {T, S <: Real}



end 