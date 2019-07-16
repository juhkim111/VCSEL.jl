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
    verbose       :: Bool = false,
    checkfrobnorm :: Bool = true
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
