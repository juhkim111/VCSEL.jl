"""
    vcselect(y, X, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Project covariate matrix `X` to null space of `X'` and select variance components 
at specified lambda by minimizing penalized negative log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]`.
Minimization is achieved via majorization-minimization (MM) algorithm. 

# Input
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty(γ = 2.0)
- `λ`: penalty strength
- `penwt`: vector of penalty weights
- `σ2`: initial values
- `maxiter`: maximum number of iterations
- `tol`: tolerance in objective value
- `verbose`: display switch

# Output
- `σ2`: minimizer
- `obj`: objevtive value at the minimizer
- `niter`: number of iterations
- `Ω`: covariance matrix evaluated at the minimizer
- `Ωinv`: precision (inverse covariance) matrix evaluated at the minimizer
"""
function vcselect( 
    yobs    :: Vector{T},
    Xobs    :: Matrix{T},
    Vobs    :: Vector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = 1.0,
    penwt   :: Vector{T} = [ones(length(V)-1); 0.0],
    σ2      :: Vector{T} = ones(length(V)),
    maxiter :: Int = 1000,
    tol     :: T = 1e-6,
    verbose :: Bool = false
    ) where {T <: Float64}

    y, V = projectToNullSpace(yobs, Xobs, Vobs)

    # call vcselect 
    σ2, obj, niters, Ω, Ωinv = vcselect(y, V; penfun=penfun, λ=λ, penwt=penwt, σ2=σ2,
                                        maxiter=maxiter, tol=tol, verbose=verbose)

    return σ2, obj, niters, Ω, Ωinv;
end

"""
    vcselect(y, V; penfun, λ, penwt, σ2, maxiter, tol, verbose)

Select variance components at specified lambda by minimizing penalized negative 
log-likelihood of variance component model. 
The objective function to minimize is
  `0.5n*log(2π) + 0.5logdet(Ω) + 0.5y'*inv(Ω)*y + λ * sum(penwt.*penfun(σ))`
where `Ω = σ2[1]*V[1] + ... + σ2[end]*V[end]`.
Minimization is achieved via majorization-minimization (MM) algorithm. 

# Input
- `y`: response vector
- `V`: vector of covariance matrices
- `penfun`: penalty function, e.g., NoPenalty() (default), L1Penalty(), MCPPenalty(γ = 2.0)
- `λ`: penalty strength
- `penwt`: vector of penalty weights
- `σ2`: initial values
- `maxiter`: maximum number of iterations
- `tol`: tolerance in objective value
- `verbose`: display switch

# Output
- `σ2`: minimizer
- `obj`: objevtive value at the minimizer
- `niter`: number of iterations
- `Ω`: covariance matrix evaluated at the minimizer
- `Ωinv`: precision (inverse covariance) matrix evaluated at the minimizer
"""
function vcselect( 
    y       :: Vector{T},
    V       :: Vector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    λ       :: T = 1.0,
    penwt   :: Vector{T} = [ones(length(V)-1); 0.0],
    σ2      :: Vector{T} = ones(length(V)),
    maxiter :: Int = 1000,
    tol  :: T = 1e-6,
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
    vcselectPath(yobs, Xobs, Vobs; penfun=NoPenalty(), penwt=[ones(length(Vobs)-1); 0.0], 
            nlambda=100, λpath=Float64[], σ2=ones(length(Vobs)), maxiter=1000, tol=1e-6)

Generate solution path of variance components along varying lambda values.

# Input  
- `yobs::Vector{Float64}`: response vector. 
- `Xobs::Matrix{Float64}`: covariate matrix.
- `Vobs::Vector{Matrix{Float64}}`: vector of covariance matrices; (V1,...,Vm).
- `penfun::Penalty`: penalty function. Default is NoPenalty().
- `penwt::Vector{Float64}`: weights for penalty term. Default is (1,1,...1,0).
- `nlambda::Int`: number of tuning parameter values. Default is 100. 
- `λpath::Vector{Float64}`: user-supplied grid of tuning parameter values. 
        If unspeficied, internally generate a grid.
- `σ2::Vector{Float64}`: initial estimates.
- `maxiter::Int`: maximum number of iteration for MM loop.
- `tol::Float64`: tolerance in objective value for MM loop.
- `verbose::Bool`: display switch. 

# Output 
- `σ2path`: solution path along varying lambda values. 
        Each column gives estimated σ2 at specific lambda.
- `objpath`: objective value path. 
- `λpath`: tuning parameter path.
"""
function vcselectPath(
    yobs    :: Vector{Float64},
    Xobs    :: Matrix{Float64},
    Vobs    :: Vector{Matrix{Float64}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: Vector{Float64} = [ones(length(Vobs)-1); 0.0],
    nlambda :: Int = 100, 
    λpath   :: Vector{Float64} = Float64[],
    σ2      :: Vector{Float64} = ones(length(Vobs)),
    maxiter :: Int = 1000,
    tol     :: Float64 = 1e-6,
    verbose :: Bool = false
    ) 

    # number of groups 
    m = length(Vobs) - 1

    ## REML: find B s.t. columns of B span the null space of X' and B'B = I
    # pivoted QR factorization of I-X(X'X)^{-1}X'
    QRfact = qr(I - Xobs * inv(cholesky(Xobs' * Xobs)) * Xobs', Val(true))
    # extract orthonormal basis of C(I-P)
    B = QRfact.Q[:, abs.(diag(QRfact.R)) .> tol] 
    # REML transformed response vector 
    y = B' * yobs 
    # REML transformed covariance matrices 
    V  = Array{Matrix{Float64}}(undef, m + 1)
    for i in 1:(m + 1)
        V[i] = B' * Vobs[i] * B  
    end  

    # make sure frobenius norm of Vi equals to 1 
    for i in 1:(m + 1)
        if norm(V[i]) != 1
            V[i] ./= norm(V[i])
        end 
    end 

    if penfun != NoPenalty() 

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ = maxlambda(y, V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nlambda)
        end 

        # initialize solution path 
        σ2path = zeros(m + 1, nlambda)
        objpath = zeros(nlambda)

        # create solution path 
        for iter in 1:length(λpath)
            λ = λpath[iter]
            σ2path[:, iter], objpath[iter], = vcselect(y, V; penfun=penfun, λ=λ, penwt=penwt,  
                        σ2=σ2, maxiter=maxiter, tol=tol, verbose=verbose)
        end

    else # if no penalty, there is no lambda grid 
        σ2path, objpath, = vcselect(y, V; penfun=penfun)
    end 

    # output 
    return σ2path, objpath, λpath 


end 