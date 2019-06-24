export maxlambda 

"""

    maxlambda(y, V; 
              penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], maxiters=500, tol=1e-8)

Find the value of λ where all σ turns 0. At any value greater than λ, all σ's are 0.

# Input 
- `y`: response vector
- `V`: vector of covariance matrices

# Keyword 
- `penfun`: penalty function, 
      possible options are NoPenalty() (default), L1Penalty(), MCPPenalty()
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `maxiters`: maximum number of iterations, default is 500
- `tol`: value below which σ is considered 0

# Output
- `maxλ`: value of λ where all coeffecients turn zero, maximum λ on the solution path 
"""
function maxlambda(
    y       :: AbstractVector{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    maxiters :: Int = 500,
    tol     :: AbstractFloat = 1e-10
    ) where {T <: Real}

    # initialize values 
    n = length(y)
    σ2_0 = dot(y, y) / n
    m = length(V) - 1

    # initialize array 
    λpath = Array{Float64}(undef, m)
    Vy = similar(y)

    # use derivative to find plausible lambda for each of `m` variance components 
    for i in eachindex(λpath)
        mul!(Vy, V[i], y) 
        λpath[i] = (1 / penwt[i]) * (-tr(V[i]) / σ2_0 +
                dot(y, Vy) / σ2_0^2)
    end
    # find maximum among m different lambdas
    tempλ = maximum(λpath)
    σ2 = zeros(length(V))
    if tempλ <= 0
	      tempλ = 30.0
    end

    # make sure all estimated σ2 are 0 at current λ
    while true
      # Step 1: obtain sigmas at tempλ
      σ2, = vcselect(y, V; penfun=penfun, λ=tempλ, penwt=penwt)
      # Step 2: if all sigmas are zero, break the loop and move to Step 3.
      maximum(view(σ2, 1:m)) < tol && break;
      # Step 2: else, multiply lambda by 1.5 and go to Step 1.
      tempλ = 1.5 * tempλ
    end

    # Step 3: Use bisection method
    iter = 1
    a = 0.5 * tempλ
    b = tempλ
    c = 0.0
    σ2_a = Array{Float64}(undef, length(V))
    σ2_b = Array{Float64}(undef, length(V))
    σ2_c = Array{Float64}(undef, length(V))

    # loop through while iteration number less than maxiters 
    while iter <= maxiters
      c = (a + b) / 2
      σ2_a, = vcselect(y, V; penfun=penfun, λ=a, penwt=penwt)
      σ2_b, = vcselect(y, V; penfun=penfun, λ=b, penwt=penwt)
      σ2_c, = vcselect(y, V, penfun=penfun, λ=c, penwt=penwt)

      if maximum(view(σ2_a, 1:m)) < tol 
        b = a
        a = b / 2
      # given that at least one σ2 at a is non-zero, if difference between
      #   σ2 at a and b are really small, break the loop
      elseif maximum(abs, view(σ2_b, 1:m) - view(σ2_a, 1:m)) < tol || (b-a) < 0.01
        break
      elseif maximum(view(σ2_c, 1:m)) > tol 
        a = c
      else
        b = c
      end
      iter = iter + 1
    end

    maxλ = b
    return maxλ, iter
end

"""

    maxlambda(Y, V; 
              penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], maxiters=500, tol=1e-8)

Find the value of λ where all σ turns 0. At any value greater than λ, all σ's are 0.

# Input 
- `Y`: response matrix
- `V`: vector of covariance matrices

# Keyword 
- `penfun`: penalty function, 
      possible options are NoPenalty() (default), L1Penalty(), MCPPenalty()
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `maxiters`: maximum number of iterations, default is 500
- `tol`: value below which σ is considered 0

# Output
- `maxλ`: value of λ where all coeffecients turn zero, maximum λ on the solution path 
"""
function maxlambda(
    Y       :: AbstractMatrix{T},
    V       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    maxiters :: Int = 500,
    tol     :: AbstractFloat = 1e-10,
    tempλ   :: AbstractFloat = 50.0
    ) where {T <: Real}

    # initialize values 
    n, d = size(Y)
    nvarcomps = length(V) 
    m = length(V) - 1

    # 
    YtY = BLAS.gemm('T', 'N', Y, Y) # Yt * Y 
    cholYtY = cholesky!(Symmetric(YtY))          
    YtYinv = inv(cholYtY)
    W = Y * YtYinv                  # Y * inv(Yt * Y)
    const1 = -n * tr(YtYinv)
    n2 = n^2

    # initialize array 
    λpath = Array{Float64}(undef, m)

    # use derivative to find plausible lambda for each of `m` variance components 
    for i in eachindex(λpath)
        λpath[i] =  const1 * tr(V[i]) + 
                n2 * tr(BLAS.gemm('T', 'N', W, V[i] * W))
    end

    # find maximum among m different lambdas
    tempλ = maximum(λpath)
    σ2 = zeros(length(V))
    if tempλ <= 0
	      tempλ = 30.0
    end

    # make sure all estimated σ2 are 0 at current λ
    while true
      # Step 1: obtain sigmas at tempλ
      Σ, = vcselect(Y, V; penfun=penfun, λ=tempλ, penwt=penwt)
      # Step 2: if all sigmas are zero, break the loop and move to Step 3.
      maximum(norm.(view(Σ, 1:m))) < tol && break;
      # Step 2: else, multiply lambda by 1.5 and go to Step 1.
      tempλ = 1.5 * tempλ
    end

    # Step 3: Use bisection method
    iter = 1
    a = 0.5 * tempλ
    b = tempλ
    c = 0.0
    Σ_a = Vector{Matrix{Float64}}(undef, nvarcomps)
    Σ_b = Vector{Matrix{Float64}}(undef, nvarcomps)
    Σ_c = Vector{Matrix{Float64}}(undef, nvarcomps)

    # loop through while iteration number less than maxiters
    while iter <= maxiters
      c = (a + b) / 2
      Σ_a, = vcselect(Y, V; penfun=penfun, λ=a, penwt=penwt)
      Σ_b, = vcselect(Y, V; penfun=penfun, λ=b, penwt=penwt)
      Σ_c, = vcselect(Y, V, penfun=penfun, λ=c, penwt=penwt)

      if maximum(norm.(view(Σ_a, 1:m))) < tol 
        b = a
        a = b / 2
      # given that at least one σ2 at a is non-zero, if difference between
      #   σ2 at a and b are really small, break the loop
      elseif maximum(abs, norm.(view(Σ_b, 1:m)) - norm.(view(Σ_a, 1:m))) < tol || (b-a) < 0.01
        break
      elseif maximum(norm.(view(Σ_c, 1:m))) > tol 
        a = c
      else
        b = c
      end
      iter = iter + 1
    end

    maxλ = b

    return maxλ, iter
end
