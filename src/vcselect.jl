"""
    vcselectpath!(vcm; penfun=NoPenalty(), penwt=[ones(nvarcomps(vcm)-1); 0], nλ=100, 
                    λpath=zeros(0), maxiters=1000, tol=1e-6, verbose=false)

Generate solution path of variance components along varying lambda values.

# Input
- `vcm`: VCModel

# Keyword 
- `penfun`: penalty function (e.g. NoPenalty(), L1Penalty(), MCPPenalty()), default is NoPenalty()
- `penwt`: weights for penalty term, default is (1,1,...1,0)
- `nλ`: number of tuning parameter values, default is 100
- `λpath`: a user supplied `λ` sequence. Typically the program computes its own `λ` 
        sequence based on `nλ`; supplying `λpath` overrides this
- `maxiter`: maximum number of iteration for MM loop, default is 1000
- `standardize`: logical flag for covariance matrix standardization, default is `true`.
    If true, `V[i]` is standardized by its Frobenius norm
- `tol`: convergence tolerance, default is `1e-6`
- `verbose`: display switch, default is false 

# Output 
- `Σ̂path`: matrix of estimated variance components at each tuning parameter `λ`,
        each column gives vector of estimated variance components `σ2` at certain `λ`
- `β̂path`: matrix of fixed parameter estimates at each `λ`
- `λpath`: sequence of `λ` values used
- `objpath`: vector of objective values at each tuning parameter `λ` 
- `niterspath`: vector of no. of iterations to convergence 
"""
function vcselectpath!(
    vcm          :: VCModel; 
    penfun       :: Penalty = NoPenalty(), 
    penwt        :: AbstractArray = [ones(nvarcomps(vcm)-1); 0],
    nλ           :: Int = 100, 
    λpath        :: AbstractArray = zeros(0), 
    maxiters     :: Int = 1000, 
    standardize  :: Bool = true, 
    tol          :: AbstractFloat = 1e-6, 
    verbose      :: Bool = false
    ) 

    # handle errors 
    @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty(), L1Penalty(), or MCPPenalty()!\n"
    @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
    @assert size(λpath, 2) <= 1 "λpath must be one-dimensional array!\n"
    @assert maxiters > 0 "maxiters should be a positive integer!\n"

    # dimension of X 
    p = ncovariates(vcm)
    d = length(vcm)

    # type 
    T = eltype(vcm.Y)

    ## generate solution path based on penalty 
    if penfun != NoPenalty()

        # create a lambda grid if not specified  
        if isempty(λpath) 
            maxλ, = maxlambda(vcm.Y, vcm.V; penfun=penfun, penwt=penwt)
            λpath = range(0, stop=maxλ, length=nλ)
        else # if lambda grid specified, make sure nlambda matches 
            nλ = length(λpath)
        end 

        # initialize arrays 
        objpath = zeros(T, nλ)
        niterspath = zeros(Int, nλ)
        if typeof(vcm.Σ[1]) <: Matrix 
            # initialize arrays
            Σ̂path = Array{Matrix{T}}(undef, nvarcomps(vcm), nλ)
            β̂path = Array{Matrix{T}}(undef, nλ)
            # solution path 
            for iter in 1:nλ
                _, objpath[iter], niterspath[iter] = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
                Σ̂path[:, iter] = vcm.Σ 
                β̂path[iter] = vcm.β
            end
        else
            # initialize arrays
            Σ̂path = zeros(T, nvarcomps(vcm), nλ)
            β̂path = zeros(T, p, nλ)
            # solution path 
            for iter in 1:nλ
                _, objpath[iter], niterspath[iter] = 
                        vcselect!(vcm; penfun=penfun, λ=λpath[iter], penwt=penwt, 
                        maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
                Σ̂path[:, iter] .= vcm.Σ
                β̂path[:, iter] .= vcm.β
            end
        end 

        return Σ̂path, β̂path, λpath, objpath, niterspath

    else # if no penalty, there is no lambda grid 
        _, objpath, niterspath = vcselect!(vcm; penfun=penfun, penwt=penwt, 
            maxiters=maxiters, tol=tol, verbose=verbose, checktype=false)
        
        return vcm.Σ, vcm.β, zeros(1), objpath, niterspath 
    end  

end 

"""
    vcselect!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

# Input 
- `vcm`: VCModel

# Keyword Argument 
- `penfun`: penalty function, default is NoPenalty()
- `λ`: tuning parameter, default is 1      
- `penwt`: penalty weights, default is [1,...1,0]
- `standardize`: logical flag for covariance matrix standardization, default is `true`.
    If true, `V[i]` is standardized by its Frobenius norm, and parameter estimates are 
    returned on the original scale
- `maxiters`: maximum number of iterations, default is 1000
- `tol`: convergence tolerance, default is `1e-6`
- `verbose`: display switch, default is false  
- `checktype`: check argument type switch, default is true

# Output 
- `vcm`: VCModel with updated `Σ` and `β` 
    Access estimates with `vcm.Σ` and `vcm.β`
- `obj`: objective value at convergence 
- `niters`: number of iterations to convergence 
"""
function vcselect!(
    vcm          :: VCModel;
    penfun       :: Penalty = NoPenalty(),
    λ            :: Real = 1.0,
    penwt        :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize  :: Bool = true, 
    maxiters     :: Int = 1000,
    tol          :: Real = 1e-6,
    verbose      :: Bool = false,
    checktype    :: Bool = true 
    ) 

    # handle errors 
    if checktype 
        @assert penfun ∈ [NoPenalty(), L1Penalty(), MCPPenalty()] "penfun must be either NoPenalty(), L1Penalty(), or MCPPenalty()!\n"
        @assert size(penwt, 2) <= 1 "penwt mut be one-dimensional array!\n"
        @assert maxiters > 0 "maxiters should be a positive integer!\n"
        # start point has to be strictly positive
        @assert all(norm.(vcm.Σ) .> 0) "starting Σ should be strictly positive or non-zero matrices"
    end 

    # update weight with reciprocal of frobenius norm 
    if standardize 
        vcm.wt .= 1 ./ norm.(vcm.V)
    end 

    # multivariate update 
    if typeof(vcm.Σ[1]) <: Matrix  # length(vcm) > 1
        _, obj, niters = mm_update_Σ!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
    
    # univariate update 
    else
        _, obj, niters = mm_update_σ2!(vcm; penfun=penfun, λ=λ, 
            penwt=penwt, maxiters=maxiters, tol=tol, verbose=verbose)
    end 

    # output 
    return vcm, obj, niters

end 

"""
    mm_update_Σ!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `Σ` using MM algorithm.
"""
function mm_update_Σ!(
    vcm       :: VCModel;
    penfun    :: Penalty = NoPenalty(),
    λ         :: Real = 1.0,
    penwt     :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize :: Bool = true, 
    maxiters  :: Int = 1000, 
    tol       :: Real = 1e-6,
    verbose   :: Bool = false 
    ) 

    # initialize algorithm 
    n, d = size(vcm)

    # working arrays 
    kron_I_one = kron(Matrix(I, d, d), ones(n)) # dn x d
   
    # initial objective value 
    updateΩ!(vcm)
    update_arrays!(vcm)
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)

    # # display 
    if verbose 
        println("iter = 0")
        println("Σ    = ", vcm.Σ)
        println("obj  = ", obj)
        #objvec = obj 
    end  

    Σtmp = deepcopy(vcm.Σ)

    ## MM loop 
    niters = 0
    for iter in 1:maxiters
        for i in eachindex(vcm.Σ)
            # if previous iterate is zero, move on to the next component
            if iszero(norm(vcm.Σ[i])) 
                continue 
            end 

            # `(kron_I_one)' * [kron(ones(d, d), V[i]) .* Ωinv] * (kron_I_one)`
            copyto!(vcm.Mndxnd, vcm.kron_ones_V[i] .* vcm.Ωinv)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', vcm.wt[i], kron_I_one, vcm.Mndxnd * kron_I_one))
        
            # add penalty unless it's the last variance component 
            if isa(penfun, L1Penalty) && i < nvarcomps(vcm) 
                penconst = λ * penwt[i] / √tr(vcm.Σ[i])
                for j in 1:d
                    vcm.Mdxd[j, j] += penconst  
                end             
            end 

            copyto!(vcm.L, cholesky!(Symmetric(vcm.Mdxd)).L)
            vcm.Linv[:] = inv(vcm.L)

            # 
            copyto!(vcm.Mdxd, vcm.Σ[i] * vcm.L)
            copyto!(vcm.Mnxd, vcm.R * vcm.Mdxd)
            copyto!(vcm.Mdxd, BLAS.gemm('T', 'N', vcm.Mnxd, vcm.V[i] * vcm.Mnxd))

            # 
            storage = eigen!(Symmetric(vcm.Mdxd))
            # if negative value, set it to 0
            @inbounds for k in 1:d
                storage.values[k] = storage.values[k] > 0 ? √storage.values[k] : 0
            end 
            copyto!(Σtmp[i], BLAS.gemm('N', 'T', 
                    storage.vectors * Diagonal(storage.values), storage.vectors))
            copyto!(Σtmp[i], BLAS.gemm('T', 'N', 
                    sqrt(vcm.wt[i]), vcm.Linv, Σtmp[i] * vcm.Linv))
        end 

        # update Σ
        clamp_diagonal!(Σtmp[end], tol, Inf)
        vcm.Σ .= Σtmp

        # update working arrays 
        updateΩ!(vcm)
        update_arrays!(vcm)

        # update objective value 
        objold = obj 
        obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
        
        # display 
        if verbose 
            println("iter = ", iter)
            println("Σ    = ", vcm.Σ)
            println("obj  = ", obj)
            #objvec = [objvec; obj]
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
        vcm.wt .= ones(nvarcomps(vcm))
    end 

    # construct final Ω matrix
    updateΩ!(vcm)
    updateΩobs!(vcm)
    updateβ!(vcm)

    # output 
    if niters == 0 
        niters = maxiters
    end 
 
    return vcm, obj, niters; 
end 

"""
    mm_update_σ2!(vcm; penfun, λ, penwt, maxiters, tol, verbose)

Update `σ2` using MM algorithm. 
"""
function mm_update_σ2!(
    vcm         :: VCModel;
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(nvarcomps(vcm)-1); 0.0],
    standardize :: Bool = true, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-6,
    verbose     :: Bool = false 
    )

    # initialize algorithm 
    n = size(vcm)[1]
    m = nvarcomps(vcm) - 1

    # initial objective value 
    updateΩ!(vcm)
    update_arrays!(vcm)
    obj = objvalue(vcm; penfun=penfun, λ=λ, penwt=penwt)
  
    # display 
    if verbose
        println("iter = 0")
        println("σ2   = ", vcm.Σ)
        println("obj  = ", obj)
        #objvec = obj 
    end    

    σ2tmp = zeros(m + 1)
  
    # MM loop 
    niters = 0
    for iter in 1:maxiters
          # update variance components
          for j in 1:m
              # move onto the next variance component if previous iterate is 0
              if iszero(vcm.Σ[j]) 
                  σ2tmp[j] = 0 
                  continue 
              end 
  
              # compute constants  
              const1 = dot(vcm.Ωinv, vcm.V[j]) # const1 = tr(Ωinv * V[j])
              mul!(vcm.Mnxd, vcm.V[j], vcm.ΩinvY)
              const2 = dot(vcm.Mnxd, vcm.ΩinvY) # const2 = y' * Ωinv * V[j] * Ωinv * y
  
              # update variance component under specified penalty 
              if !isa(penfun, NoPenalty) 
                    penstrength = λ * penwt[j]
                    # L1 penalty 
                    if isa(penfun, L1Penalty)  
                        σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + 
                                penstrength / (vcm.wt[j] * sqrt(vcm.Σ[j]))))
                    # MCP penalty 
                    elseif isa(penfun, MCPPenalty) 
                        if vcm.Σ[j] <= (penfun.γ * λ)^2
                            σ2tmp[j] = vcm.Σ[j] * √(const2 / (const1 + 
                                (λ / sqrt(vcm.Σ[j]) - 1 / penfun.γ) * (1 / vcm.wt[j])))
                        else 
                            σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
                        end 
                    end 
              # update variance component under no penalty 
              elseif isa(penfun, NoPenalty)
                σ2tmp[j] = vcm.Σ[j] * √(const2 / const1)  
              end
  
          end # end of for loop over j

          # update last variance component and Ω
          σ2tmp[end] = vcm.Σ[end] *  √(dot(vcm.ΩinvY, vcm.ΩinvY) / tr(vcm.Ωinv))
          σ2tmp[end] = clamp(σ2tmp[end], tol, Inf)

          vcm.Σ .= σ2tmp

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
              println("obj  = ", obj)
              #objvec = [objvec; obj] 
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
            vcm.wt .= ones(m + 1)
      end 
  
      # construct Ω matrix 
      updateΩ!(vcm)
      updateΩobs!(vcm)
      updateβ!(vcm)
  
      # output
      if niters == 0
        niters = maxiters
      end
   
      # 
      return vcm, obj, niters 

end 

"""
    vcselect(Y, V; penfun=NoPenalty(), λ=1.0, penwt=[ones(length(V)-1); 0.0],
                standardize=true, maxiters=1000, tol=1e-6, verbose=false, checktype=true)

"""
function vcselect(
    Y           :: AbstractVecOrMat{T},
    V           :: AbstractVector{Matrix{T}};
    penfun      :: Penalty = NoPenalty(),
    λ           :: Real = 1.0,
    penwt       :: AbstractVector = [ones(length(V)-1); 0.0],
    standardize :: Bool = true, 
    maxiters    :: Int = 1000,
    tol         :: Real = 1e-6,
    verbose     :: Bool = false,
    checktype   :: Bool = true 
    ) where {T <: Real}

    vcmtmp = VCModel(Y, V)
    _, obj, niters = vcselect!(vcmtmp; penfun=penfun, λ=λ, penwt=penwt, standardize=standardize,
            maxiters=maxiters, tol=tol, verbose=verbose, checktype=checktype)

    return vcmtmp.Σ, obj, niters

end 