"""

    maxlambda(y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], maxiter=500, tol=1e-8)

Find the value of λ where all σ turns 0. At any value greater than λ, all σ's are 0.

# Arguments
- `y::Vector{Float64}`: response vector
- `V::Vector{Matrix{Float64}}`: vector of covariance matrices
- `penfun::Penalty`: penalty function. 
            Possible options are NoPenalty() (default), L1Penalty(), MCPPenalty().
- `penwt::Vector{Float64}`: weight vector
- `maxiter::Int`: maximum number of iterations. Default is 500. 
- `tol::Float64`: value below which sigma is considered 0

# Output
- `maxλ::Float64`: value of λ where all coeffecients turn zero (i.e. smallest λ with all zero coefficients)
"""
function maxlambda(
    y       :: Vector{Float64},
    V       :: Vector{Matrix{Float64}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: Vector{Float64} = [ones(length(V)-1); 0.0],
    maxiter :: Int = 500,
    tol     :: Float64 = 1e-8
    ) 

    # initialize values 
    n = length(y)
    σ02 = dot(y, y) / n
    m = length(V) - 1

    # initialize array 
    λpath = Array{Float64}(undef, m)

    # use derivative to find plausible lambda for each of `m`variance components 
    for i in eachindex(λpath)
        Vy = V[i] * y
        λpath[i] = (1 / penwt[i]) * (-tr(V[i]) / (2 * σ02) +
                dot(y, Vy) / (2 * σ02^2))
    end

    # find maximum lambda among m different lambdas
    tempλ = maximum(λpath)
    sigmas = zeros(length(V))
    if tempλ <= 0
	      tempλ = 30.0
    end

    # make sure all estimated sigmas are 0 at current λ
    while true
      # Step 1: obtain sigmas at tempλ
      sigmas, = vcmm(y, V; penfun=penfun, λ=tempλ, penwt=penwt)
      # Step 2: if all sigmas are zero, break the loop and move to Step 3.
      all(sigmas[1:end-1] .< tol) && break;
      # Step 2: else, multiply lambda by 1.5 and go to Step 1.
      tempλ = 1.5 * tempλ
    end

    # Step 3: Use bisection method
    iter = 1
    a = 0.5 * tempλ
    b = tempλ
    c = 0.0
    sigmas_a = Array{Float64}(undef, length(V))
    sigmas_b = Array{Float64}(undef, length(V))
    sigmas_c = Array{Float64}(undef, length(V))

    # loop through while iteration number less than maxiter 
    while iter <= maxiter
      c = (a + b) / 2
      sigmas_a, = vcmm(y, V; penfun=penfun, λ=a, penwt=penwt)
      sigmas_b, = vcmm(y, V; penfun=penfun, λ=b, penwt=penwt)
      sigmas_c, = vcmm(y, V, penfun=penfun, λ=c, penwt=penwt)

      if all(sigmas_a[1:end-1] .< tol)
        b = a
        a = b / 2
      # given that at least one sigma at a is non-zero, if difference between
      #   sigmas at a and b are really small, break the loop
      elseif maximum(abs, sigmas_b[1:end-1] - sigmas_a[1:end-1]) < tol || (b-a) < 0.01
        break
      elseif any(sigmas_c[1:end-1] .> tol)
        a = c
      else
        b = c
      end
      iter = iter + 1
    end

    maxλ = b
    return maxλ
end
