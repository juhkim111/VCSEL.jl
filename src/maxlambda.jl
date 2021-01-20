"""

    maxlambda(y, V; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], maxiters=500, 
              standardize=false, tol=1e-8)

Find the value of λ where all σ turns 0. At any value greater than λ, all σ's are 0.

# Input 
- `y`: response vector
- `V`: vector of covariance matrices

# Keyword 
- `penfun`: penalty function, possible options are `L1Penalty()` (default), `MCPPenalty()`, 
    and `NoPenalty()`. If `penfun=NoPenalty()`, 0 is returned
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `standardize`: logical flag for covariance matrix standardization, default is `false`.
    If true, `V[i]` and `Vint[i]` is standardized by its Frobenius norm, and parameter 
    estimates are returned on the original scale
- `maxiters`: maximum number of iterations, default is 500
- `tol`: value below which estimate is considered zero

# Output
- `maxλ`: value of λ where all coeffecients turn zero, maximum λ on the solution path 
"""
function maxlambda(
    y           :: AbstractVector{T},
    V           :: AbstractVector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    penwt       :: AbstractVector{T} = [ones(T, length(V)-1); zero(T)],
    standardize :: Bool = false, 
    maxiters    :: Int = 500,
    tol         :: AbstractFloat = 1e-8
    ) where {T <: Real}

    # if no penalty, return 0 
    if isa(penfun, NoPenalty)
      return zero(T), 0
    end 

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
      σ2, = vcselect(y, V; penfun=penfun, λ=tempλ, penwt=penwt, standardize=standardize)
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
      σ2_a, = vcselect(y, V; penfun=penfun, λ=a, penwt=penwt, standardize=standardize)
      σ2_b, = vcselect(y, V; penfun=penfun, λ=b, penwt=penwt, standardize=standardize)
      σ2_c, = vcselect(y, V, penfun=penfun, λ=c, penwt=penwt, standardize=standardize)

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
    findmaxλ(Y, G; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
              maxiters=500, tol=1e-8)

Find the value of λ where all Σ turns 0. At any value greater than λ, all Σ's are 0.

# Input 
- `Y`: response matrix
- `V`: vector of covariance matrices

# Keyword 
- `penfun`: penalty function, possible options are `L1Penalty()` (default), `MCPPenalty()`, 
    and `NoPenalty()`. If `penfun=NoPenalty()`, 0 is returned
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `standardize`: logical flag for covariance matrix standardization, default is `false`.
    If true, `V[i]` and `Vint[i]` is standardized by its Frobenius norm, and parameter 
    estimates are returned on the original scale
- `maxiters`: maximum number of iterations, default is 500
- `tol`: value below which estimate is considered zero

# Output
- `maxλ`: value of λ where all coeffecients turn zero, maximum λ on the solution path 
"""
function findmaxλ(
    Y       :: AbstractVecOrMat{T},
    G       :: AbstractVector{Matrix{T}};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, length(G)-1); zero(T)],
    #standardize :: Bool = false, 
    maxiters :: Int = 500,
    tol     :: AbstractFloat = 1e-8,
    tempλ   :: AbstractFloat = 50.0,
    Mdxd    :: AbstractMatrix{T} = Matrix{T}(undef, size(Y, 2), size(Y, 2)),
    Mnxd    :: AbstractMatrix{T} = Matrix{T}(undef, size(Y, 1), size(Y, 2))
    ) where {T <: Real}

    # if no penalty, return 0 
    if isa(penfun, NoPenalty)
      return zero(T), 0
    end 

    # initialize values 
    n, d = size(Y)
    m = length(G) 

    # 
    BLAS.syrk!('U', 'T', true, Y, false, Mdxd)
    chol = cholesky!(Symmetric(Mdxd))          
    cholinv = inv(chol)
    const1 = -n * tr(cholinv)
    mul!(Mnxd, Y, cholinv) # n * d

    # initialize array 
    potentialλs = Array{T}(undef, m)

    # use derivative to find plausible lambda for each of `m` variance components 
    for i in eachindex(potentialλs)
      BLAS.syrk!('U', 'T', true, G[i]' * Mnxd, false, Mdxd)
      potentialλs[i] =  const1 * tr(G[i] * G[i]') + n^2 * tr(Mdxd)
    end
    println("potentialλs=", potentialλs)

    # find maximum among m different lambdas
    tempλ = maximum(potentialλs)
    if tempλ <= 0
	      tempλ = 30.0
    end
    println("tempλ=", tempλ)

    # make sure all estimated σ2 are 0 at current λ
    while true
      # Step 1: obtain sigmas at tempλ
      Σ, = vcselect(Y, G; penfun=penfun, λ=tempλ, penwt=penwt)
      # Step 2: if all sigmas are zero, break the loop and move to Step 3.
      maximum(tr.(view(Σ, 1:m))) < tol && break;
      # Step 2: else, multiply lambda by 1.5 and go to Step 1.
      tempλ = 1.5 * tempλ
    end
    println("tempλ=", tempλ)

    # Step 3: Use bisection method
    iter = 1
    a = 0.5 * tempλ
    b = tempλ
    c = 0.0
    Σ_a = Vector{Matrix{T}}(undef, nvarcomps)
    Σ_b = Vector{Matrix{T}}(undef, nvarcomps)
    Σ_c = Vector{Matrix{T}}(undef, nvarcomps)

    # loop through while iteration number less than maxiters
    while iter <= maxiters
      c = (a + b) / 2
      Σ_a, = vcselect(Y, G; penfun=penfun, λ=a, penwt=penwt)
      Σ_b, = vcselect(Y, G; penfun=penfun, λ=b, penwt=penwt)
      Σ_c, = vcselect(Y, G; penfun=penfun, λ=c, penwt=penwt)

      if maximum(tr.(view(Σ_a, 1:m))) < tol 
        b = a
        a = b / 2
      # given that at least one σ2 at a is non-zero, if difference between
      #   σ2 at a and b are really small, break the loop
      elseif maximum(abs, tr.(view(Σ_b, 1:m)) - tr.(view(Σ_a, 1:m))) < tol || (b-a) < 0.01
        break
      elseif maximum(tr.(view(Σ_c, 1:m))) > tol 
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
    findmaxλ(vcm; penfun=NoPenalty(), penwt=[ones(length(V)-1); 0.0], 
              maxiters=500, tol=1e-8)

Find the value of λ where all Σ turns 0. At any value greater than λ, all Σ's are 0.

# Input 
- `vcm`: VCModel 

# Keyword 
- `penfun`: penalty function, possible options are `L1Penalty()` (default), `MCPPenalty()`, 
    and `NoPenalty()`. If `penfun=NoPenalty()`, 0 is returned
- `penwt`: vector of penalty weights, default is (1,1,...1,0)
- `standardize`: logical flag for covariance matrix standardization, default is `false`.
    If true, `V[i]` and `Vint[i]` is standardized by its Frobenius norm, and parameter 
    estimates are returned on the original scale
- `maxiters`: maximum number of iterations, default is 500
- `tol`: value below which estimate is considered zero

# Output
- `maxλ`: value of λ where all coeffecients turn zero, maximum λ on the solution path 
"""
function findmaxλ(
    vcm     :: VCModel{T};
    penfun  :: Penalty = NoPenalty(),
    penwt   :: AbstractVector{T} = [ones(T, nvarcomps(vcm)-1); zero(T)],
    #standardize :: Bool = false, 
    maxiters :: Int = 500,
    tol     :: AbstractFloat = 1e-6,
    tempλ   :: AbstractFloat = 0.0
    ) where {T <: Real}

    println("------- starting findmaxλ -------")
    println("penfun = ", penfun, ", maxiters = ", maxiters, ", tol = ", tol)
    # 
    vcmtmp = deepcopy(vcm)
    Σinit = deepcopy(vcm.Σ)

    # if no penalty, return 0 
    if isa(penfun, NoPenalty)
      return zero(T), 0
    end 

    # initialize values 
    n, d = size(vcmtmp)[1], size(vcmtmp)[2]
    m = nvarcomps(vcmtmp) - 1 

    # 
    if tempλ <= 0 
      BLAS.syrk!('U', 'T', true, reshape(vcmtmp.vecY, n, d), false, vcmtmp.Mdxd)
      chol = cholesky!(Symmetric(vcmtmp.Mdxd))          
      vcmtmp.Mdxd[:] = inv(chol)
      const1 = -n * tr(vcmtmp.Mdxd)
      mul!(vcmtmp.Mnxd, reshape(vcmtmp.vecY, n, d), vcmtmp.Mdxd) # n * d

      # initialize array 
      potentialλs = Array{T}(undef, m)

      # use derivative to find plausible lambda for each of `m` variance components 
      for i in eachindex(potentialλs)
        BLAS.syrk!('U', 'T', true, vcmtmp.G[i]' * vcmtmp.Mnxd, false, vcmtmp.Mdxd)
        potentialλs[i] =  const1 * tr(vcmtmp.G[i] * vcmtmp.G[i]') + n^2 * tr(vcmtmp.Mdxd)
      end

      # find maximum among m different lambdas
      tempλ = maximum(potentialλs)
      if tempλ <= 0
          tempλ = 30.0
      end
    end 

    # make sure all estimated σ2 are 0 at current λ
    while true
      # Step 1: obtain sigmas at tempλ
      vcselect!(vcmtmp; penfun=penfun, λ=tempλ, penwt=penwt)
      # Step 2: if all sigmas are zero, break the loop and move to Step 3.
      maximum(tr.(view(vcmtmp.Σ, 1:m))) < tol && break;
      resetModel!(vcmtmp, Σinit)
      # Step 2: else, multiply lambda by 1.5 and go to Step 1.
      tempλ = 1.5 * tempλ
    end

    # Step 3: Use bisection method
    #resetModel!(vcmtmp, Σinit)
    iter = 1
    a = 0.5 * tempλ
    b = tempλ
    c = 0.0
    Σ_a = deepcopy(Σinit) 
    Σ_b = deepcopy(Σinit) 
    Σ_c = deepcopy(Σinit) 

    # loop through while iteration number less than maxiters
    while iter <= maxiters
      println("iter : ", iter)
      println("a=", a, " b=", b, " c=", c)
      c = (a + b) / 2

      resetModel!(vcmtmp, Σinit)     
      vcselect!(vcmtmp; penfun=penfun, λ=a, penwt=penwt, checktype=false)
      copyto!(Σ_a, vcmtmp.Σ)
      resetModel!(vcmtmp, Σinit)
      vcselect!(vcmtmp; penfun=penfun, λ=b, penwt=penwt, checktype=false)
      copyto!(Σ_b, vcmtmp.Σ)
      resetModel!(vcmtmp, Σinit)
      vcselect!(vcmtmp; penfun=penfun, λ=c, penwt=penwt, checktype=false)
      copyto!(Σ_c, vcmtmp.Σ)
      

      if maximum(tr.(view(Σ_a, 1:m))) < tol 
        b = a
        a = b / 2
      # given that at least one σ2 at a is non-zero, if difference between
      #   σ2 at a and b are really small, break the loop
      elseif maximum(abs, tr.(view(Σ_b, 1:m)) - tr.(view(Σ_a, 1:m))) < tol || (b-a) < 0.01
        break
      elseif maximum(tr.(view(Σ_c, 1:m))) > tol 
        a = c
      else
        b = c
      end
      iter = iter + 1
    end

    maxλ = b
    println("maxλ = ", maxλ)
    println("------ findmaxλ finished -------")
    return maxλ, iter
end