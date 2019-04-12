"""
    vcmm(y, V; penfun, λ, penwt, σ2, maxiter, tolfun, verbose)

Minimizes penalized negative log-likelihood of variance component model using 
MM algorithm.
The objective function is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]`.

# Input
- `y`: response vector
- `V`: vector of covariance matrices
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty(γ = 2.0)
- `λ`: penalty strength
- `penwt`: vector of penalty weights
- `σ2`: initial values
- `maxiter`: maximum number of iterations
- `tolfun`: tolerance in objective value
- `verbose`: display switch

# Output
- `σ2`: minimizer
- `obj`: objevtive value at the minimizer
- `niter`: number of iterations
- `Ω`: covariance matrix evaluated at the minimizer
- `Ωinv`: precision (inverse covariance) matrix evaluated at the minimizer
"""
function vcmm( 
    y       :: Vector{T},
    V       :: Vector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = 1.0,
    penwt   :: Vector{T} = [ones(length(V)-1); 0.0],
    σ2      :: Vector{T} = ones(length(V)),
    maxiter :: Int = 1000,
    tolfun  :: T = 1e-6,
    verbose :: Bool = false
    ) where {T <: Float64}

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
                        if σ2[j] <= sqrt(penfun.γ * λ)
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
        if abs(obj - objold) < tolfun * (abs(objold) + 1)
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
