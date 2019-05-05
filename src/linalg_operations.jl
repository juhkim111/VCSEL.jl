export nullprojection, fixedeffects

"""
    nullprojection(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate matrix, response `y` is projected onto the null space of transpose(X) 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I/√n)
note that each V[i] needs to have frobenius norm 1, and that V[end] should be 
identity matrix divided by √n   

# Ouptut 
- `ynew`: projected response vector
- `Vnew`: projected vector of covariance matrices, 
frobenius norm of `V[i]` equals to 1 for all `i`
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    y    :: AbstractVector{T},
    X    :: AbstractMatrix{T},
    V    :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    # basis of nullspace of transpose(X), `N(X')`
    Xt = similar(X')
    transpose!(Xt, X)
    B = nullspace(Xt)

    # projected response vector 
    ynew = B' * y 

    # dimension of null space 
    s = size(B, 2) 

    # no. of variance components subject to selection 
    m = length(V) - 1

    # transformed covariance matrices 
    Vnew = Vector{Matrix{Float64}}(undef, m + 1)
    #Vnew = similar(V)
    Vnew[end] = Matrix{eltype(B)}(I, s, s) ./ √s
    tmp = zeros(size(X, 1), s)
    for i in 1:m
        mul!(tmp, V[i], B)
        Vnew[i] = BLAS.gemm('T', 'N', B, tmp)
        # divide by its frobenius norm  
        Vnew[i] ./= norm(Vnew[i])
    end 

    # output 
    return ynew, Vnew, B 

end 

"""
    fixedeffects(y, X, V, σ2)

Estimate fixed effects using REML estimate of variance components.
Estimate of beta is 
    `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
where `Ω` being `∑ σ2[i] * V[i]` where `σ2` is the REML estimate.

# Input
- `y`: response vector
- `X`: covariate matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
    note that V[end] should be identity matrix
- `σ2`: REML estimate of variance components 

# Output 
- `β`: fixed effects estimate
"""
function fixedeffects( 
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

    β = fixedeffects(y, X, Ω)

    return β

end 
"""
    fixedeffects(y, X, Ω)

Estimate fixed effects using REML estimate of variance components.
Estimate of beta is 
    `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
where `Ω` being `∑ σ2[i] * V[i]` where `σ2` is the REML estimate.

# Input
- `y`: response vector
- `X`: covariate matrix 
- `Ω`: overall covariance matrix constructed using REML estimate of variance components or
cholesky factorization of the overall covariance matrix 

# Output 
- `β`: fixed effects estimate Ω supplied is a Cholesky object, default is false
"""
function fixedeffects( 
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
    kronaxpy!(A, X, Y)

Overwrites `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)` but more efficient.
"""
function kronaxpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T}, 
    Y :: AbstractVecOrMat{T}
    ) where {T <: Real}

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