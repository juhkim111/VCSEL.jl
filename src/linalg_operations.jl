export nullprojection, getfixedeffects, kronaxpy!, clamp_diagonal!

"""
    objvalue(vcm; penfun, λ, penwt)

Calculate objective value, i.e. negative log-likelihood of [`VCModel`](@ref) instance plus
penalty terms. 

# Input 
- `vcm`: VCModel

# Keyword Argument 
- `penfun`: penalty function (e.g. NoPenalty(), L1Penalty(), MCPPenalty()), 
        default is NoPenalty()
- `λ`: tuning parameter, default is 1
- `penwt`: penalty weight, default is (1,...1,0)

# Output 
- `obj`: objective value 

"""
function objvalue(
    vcm    :: VCModel;
    penfun :: Penalty = NoPenalty(),
    λ      :: T = one(T),
    penwt  :: AbstractVector{T} = [ones(T, nvarcomps(vcm)-1); zero(T)]
    ) where {T <: Real}
   
    obj = logdet(vcm.ΩcholL) + (1 // 2) * dot(vcm.vecY, vcm.ΩinvY)
    obj += (1 // 2) * prod(size(vcm)) * log(2π)

    # add penalty term 
    if !isa(penfun, NoPenalty)
        pen = 0.0
        for j in 1:(nvarcomps(vcm) - 1)
            pen += penwt[j] * value(penfun, √tr(vcm.Σ[j]))
        end 
        obj += λ * pen
    end 

    return obj 
end 

"""
    nullprojection(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector/matrix to be transformed
- `X`: covariate matrix, response `y` is projected onto the null space of transpose(X) 
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note `V[end]` should be identity matrix or identity matrix divided by √n

# Ouptut 
- `ynew`: projected response vector/matrix
- `Vnew`: projected vector of covariance matrices
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    y    :: AbstractVecOrMat{T},
    X    :: AbstractVecOrMat{T},
    V    :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    if isempty(X)
        return y, V, X
    else
        # basis of nullspace of transpose(X), `N(X')`
        Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
        transpose!(Xt, X)
        B = nullspace(Xt)

        # projected response vector 
        ynew = B' * y 

        # dimension of null space 
        s = size(B, 2) 

        # no. of variance components subject to selection 
        nvarcomps = length(V) 

        # transformed covariance matrices 
        Vnew = Vector{Matrix{Float64}}(undef, nvarcomps)
        tmp = zeros(size(X, 1), s)
        for i in 1:(nvarcomps - 1)
            mul!(tmp, V[i], B)
            Vnew[i] = BLAS.gemm('T', 'N', B, tmp)
            # divide by its frobenius norm  
            #Vnew[i] ./= norm(Vnew[i])
        end 
        Vnew[end] = Matrix{eltype(B)}(I, s, s)

        # output 
        return ynew, Vnew, B 
    end 

end 

"""
    nullprojection(y, X, V1, V2)

Project `y` to null space of `transpose(X)` and transform `V1` and `V2` accordingly.
# Input 
- `y`: response vector to be transformed. 
- `X`: covariate vector or matrix, response `y` is projected onto the null space of transpose(X) 
- `V1`: vector of covariance matrices, (V1[1],V1[2],...,V1[m],I)
- `V2`: vector of covariance matrices, (V2[1],V2[2],...,V2[m]) 

# Ouptut 
- `ynew`: projected response vector
- `Vnew1`: projected vector of covariance matrices
- `Vnew2`: projected vector of covariance matrices
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    y  :: AbstractVecOrMat{T},
    X  :: AbstractVecOrMat{T},
    V1 :: AbstractVector{Matrix{T}},
    V2 :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    if isempty(X)
        return y, V1, V2
    else 
        # basis of nullspace of transpose(X), `N(X')`
        Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
        transpose!(Xt, X)
        B = nullspace(Xt)

        # projected response vector 
        ynew = B' * y 

        # dimension of null space 
        s = size(B, 2) 

        # no. of variance components subject to selection 
        m = length(V2) 

        # transformed covariance matrices 
        Vnew1 = Vector{Matrix{Float64}}(undef, m + 1)
        Vnew2 = Vector{Matrix{Float64}}(undef, m)

        tmp = zeros(size(X, 1), s)
        for i in 1:m
            mul!(tmp, V1[i], B)
            Vnew1[i] = BLAS.gemm('T', 'N', B, tmp)
            mul!(tmp, V2[i], B)
            Vnew2[i] = BLAS.gemm('T', 'N', B, tmp)
            # # divide by its frobenius norm  
            # Vnew1[i] ./= norm(Vnew1[i])
            # Vnew2[i] ./= norm(Vnew2[i])
        end 
        Vnew1[end] = Matrix{eltype(B)}(I, s, s) 

        # output 
        return ynew, Vnew1, Vnew2, B 
    end 

end 

"""
    updateβ!(vcm)

Estimate fixed effects using REML estimate of variance components.
Estimate of beta is 
    `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
    `beta = pinv()`
where `Ω` being `∑ σ2[i] * V[i]` or `∑ Σ[i] ⊗ V[i]` where `σ2` or `Σ` are REML estimates.

# Input
- `vcm`: VCModel

# Output 
- `vcm.β`: updated fixed parameter estimates 
"""
function updateβ!( 
    vcm :: VCModel
    ) 

    # quick return if there is no mean parameters
    isempty(vcm.β) && return vcm.β

    # 
    Ωchol = cholesky(Symmetric(vcm.Ωobs))
    # 
    d = length(vcm)
    p = size(vcm.Xobs, 2)
    kron_I_X = kron(Matrix(I, d, d), vcm.Xobs)
    XtΩinvX = BLAS.gemm('T', 'N', kron_I_X, Ωchol \ kron_I_X)
    β = BLAS.gemv('T', kron_I_X, Ωchol \ vec(vcm.Yobs))
    β = pinv(XtΩinvX) * β
    if d == 1 
        vcm.β .= β
    else 
        vcm.β .= reshape(β, p, d)
    end 
    vcm.β
end

"""
    getfixedeffects(y, X, V, σ2)

Estimate fixed effects using REML estimate of variance components.
Estimate of beta is 
    `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
where `Ω` being `∑ σ2[i] * V[i]` where `σ2` is the REML estimate.

# Input
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default   
- `σ2`: REML estimate of variance components 

# Output 
- `β`: fixed effects estimate
"""
function getfixedeffects( 
    y   :: AbstractVector{T},
    X   :: AbstractMatrix{T},
    V   :: AbstractVector{Matrix{T}},
    σ2  :: AbstractVector{T}
    ) where {T <: Real}

    # update Ω with estimated variance components 
    Ω = zeros(T, size(V[1]))
    for i in eachindex(σ2)
        if iszero(σ2[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) # Ω .+= σ2[i] * V[i]
        end 
    end 

    β = getfixedeffects(y, X, Ω)

    return β

end 
"""
    fixedeffects(Y, X, V, Σ)

Estimate fixed effects using REML estimate of variance components, given `Y` is a 
multivariate response. Let dimensions of `Y` and `X` be `nxd` and `nxp`, respectively. 
Then fixed effects parameter `β` would be `pxd` matrix. 
Estimate of `β` is obtained  
    `vec(β̂) = pinv((I ⊗ X)' * Ωinv * (I ⊗ X))((I ⊗ X)' * Ωinv * vec(Y))`
where `pinv` indicates Moore-Penrose pseudoinverse and 
`Ω` denotes `sum(Σ[i] * V[i])` where `Σ` is the REML estimates.

# Input
- `Y`: response matrix
- `X`: covariate matrix 
- `V`: vector of covariance matrices, `(V[1],V[2],...,V[m],I)`
    note (1) `V[end]` should be identity matrix or identity matrix divided by √n
    note (2) each `V[i]` needs to have frobenius norm 1, 
            if not, `vcselect` internally divides each `V[i]` by its frobenius norm by default   
- `Σ`: REML estimate of variance components 

# Output 
- `β`: fixed effects estimate
"""
function getfixedeffects( 
    Y   :: AbstractMatrix{T},
    X   :: AbstractMatrix{T},
    V   :: AbstractVector{Matrix{T}},
    Σ   :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    # update Ω with estimated variance components 
    n, d = size(Y)
    Ω = zeros(T, n*d, n*d)
    for j in eachindex(Σ)
        if isapprox(Σ[j], zeros(d, d))
            continue 
        end 
        kronaxpy!(Σ[j], V[j], Ω)
    end  

    # fixed effects estimates 
    β = getfixedeffects(Y, X, Ω)

    return β

end 
"""
    fixedeffects(y, X, Ω)

Estimate fixed effects using REML estimate of variance components, given `y` is a 
univariate response. Estimate of beta is 
    `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
where `Ω` being `∑ σ2[i] * V[i]` where `σ2` is the REML estimates.

# Input
- `y`: response vector
- `X`: covariate matrix 
- `Ω`: overall covariance matrix constructed using REML estimate of variance components or
cholesky factorization of the overall covariance matrix 

# Output 
- `β`: fixed effects estimate 
"""
function getfixedeffects( 
    y   :: AbstractVector{T},
    X   :: AbstractMatrix{T},
    Ω   :: Union{AbstractMatrix{T}, Cholesky}
    ) where {T <: Real}

    # if not cholesky factorized, perform cholesky 
    if typeof(Ω) <: Cholesky
        Ωchol = Ω
    else
        Ωchol = cholesky(Symmetric(Ω))
    end 

    # estimate fixed effects: pinv(X'*Ωinv*X)(X'*Ωinv*y)
    XtΩinvX = BLAS.gemm('T', 'N', X, Ωchol \ X)
    β = BLAS.gemv('T', X, Ωchol \ y) # overwriting Ωinv with X'*Ωinv
    β = pinv(XtΩinvX) * β

    return β

end 
"""
    fixedeffects(Y, X, Ω)

Estimate fixed effects using REML estimate of variance components, given `Y` is a 
multivariate response. Let dimensions of `Y` and `X` be `nxd` and `nxp`, respectively. 
Then fixed effects parameter `β` would be `pxd` matrix. 
Estimate of `β` is obtained  
    `vec(β̂) = pinv((I ⊗ X)' * Ωinv * (I ⊗ X))((I ⊗ X)' * Ωinv * vec(Y))`
where `pinv` indicates Moore-Penrose pseudoinverse and 
`Ω` denotes `sum(Σ[i] * V[i])` where `Σ` is the REML estimates.

# Input
- `Y`: response matrix
- `X`: covariate matrix 
- `Ω`: overall covariance matrix constructed using REML estimate of variance components or
    cholesky factorization of the overall covariance matrix 

# Output 
- `β`: fixed effects estimate 
"""
function getfixedeffects( 
    Y   :: AbstractMatrix{T},
    X   :: AbstractMatrix{T},
    Ω   :: Union{AbstractMatrix{T}, Cholesky}
    ) where {T <: Real}

    # if not cholesky factorized, perform cholesky 
    if typeof(Ω) <: Cholesky
        Ωchol = Ω
    else
        Ωchol = cholesky(Symmetric(Ω))
    end 

    # dimension 
    n, p = size(X)
    d = size(Y, 2)

    # estimate fixed effects
    kron_I_X = kron(Matrix(I, d, d), X)
    XtΩinvX = BLAS.gemm('T', 'N', kron_I_X, Ωchol \ kron_I_X)
    β = BLAS.gemv('T', kron_I_X, Ωchol \ vec(Y)) 
    β = pinv(XtΩinvX) * β

    return reshape(β, p, d)

end 
"""
    kronaxpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)` but more efficient.
"""
function kronaxpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T}, 
    Y :: AbstractVecOrMat{T}
    ) where {T}

    # retrieve matrix sizes
    m, n = size(A, 1), size(A, 2)
    p, q = size(X, 1), size(X, 2)
    # loop over (i,j) blocks of Y
    irange, jrange = 1:p, 1:q
    @inbounds for j in 1:n, i in 1:m
        a = A[i, j]
        irange = ((i - 1) * p + 1):(i * p)
        jrange = ((j - 1) * q + 1):(j * q)
        Yij = view(Y, irange, jrange)  # view of (i, j)-block
        @simd for k in eachindex(Yij)
        Yij[k] += a * X[k]
        end
    end
    Y
end
"""
    kronaxpy!(a, X, Y)

Overwrite `Y` with `a*X + Y`. Same as `axpy!(a, X, Y)`.
"""
function kronaxpy!(
    a :: T,
    X :: AbstractVecOrMat{T},
    Y :: AbstractVecOrMat{T}
    ) where {T}

    axpy!(a, X, Y)
    Y
end 

"""
    clamp_diagonal!(A, lo, hi)

Clamp the diagonal entries of matrix `A` to `[lo, hi]`.
"""
function clamp_diagonal!(
    A  :: Matrix{T}, 
    lo :: T, 
    hi :: T
    ) where {T}

    @inbounds @simd for i in 1:minimum(size(A))
        A[i, i] = clamp(A[i, i], lo, hi)
    end
    A
end