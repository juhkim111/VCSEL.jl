export nullprojection, kronaxpy!, clamp_diagonal!

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
    λ      :: Real = 1.0,
    penwt  :: AbstractVector{T} = [ones(T, nvarcomps(vcm)-1); zero(T)]
    ) where {T <: Real}
   
    obj = logdet(vcm.ΩcholL) + (1 // 2) * dot(vcm.vecY, vcm.ΩinvY)
    obj += (1 // 2) * prod(size(vcm)) * log(2π)

    # add penalty term 
    if !isa(penfun, NoPenalty)
        pen = 0.0
        # L1Penalty
        if isa(penfun, L1Penalty)
            # add penalty term 
            for j in 1:(nvarcomps(vcm) - 1)
                pen += penwt[j] * √tr(vcm.Σ[j])
            end 
            obj += λ * pen
        # MCP penalty 
        elseif isa(penfun, MCPPenalty)
            const1 = penfun.γ * λ
            const2 = (1/2) * const1 * λ
            # add penalty term 
            for j in 1:(nvarcomps(vcm) - 1)
                if √tr(vcm.Σ[j]) < const1
                    pen += λ * √tr(vcm.Σ[j]) - tr(vcm.Σ[j]) / (2*penfun.γ)
                else
                    pen += const2
                end 
            end 
            obj += pen 
        end 
    end 

    return obj 
end 

"""
    objvalue(vcm; penfun, λ, penwt)

Calculate objective value, i.e. negative log-likelihood of [`VCintModel`](@ref) instance 
plus penalty terms. 

# Input 
- `vcm`: VCintModel

# Keyword Argument 
- `penfun`: penalty function (e.g. NoPenalty(), L1Penalty(), MCPPenalty()), 
        default is NoPenalty()
- `λ`: tuning parameter, default is 1
- `penwt`: penalty weight, default is (1,...1,0)

# Output 
- `obj`: objective value    
"""
function objvalue(
    vcm    :: VCintModel;
    penfun :: Penalty = NoPenalty(),
    λ      :: Real = 1.0,
    penwt  :: AbstractVector{T} = [ones(T, ngroups(vcm)); zero(T)]
    ) where {T <: Real}

    obj = logdet(vcm.ΩcholL) + (1 // 2) * dot(vcm.vecY, vcm.ΩinvY)
    obj += (1 // 2) * size(vcm)[1] * log(2π)

    # add penalty term 
    if !isa(penfun, NoPenalty)
        pen = 0.0
        # L1Penalty
        if isa(penfun, L1Penalty)
            # add penalty term 
            for j in 1:ngroups(vcm)
                pen += penwt[j] * √(vcm.Σ[j] + vcm.Σint[j])
            end 
            obj += λ * pen
        # MCP penalty 
        elseif isa(penfun, MCPPenalty)
            const1 = penfun.γ * λ
            const2 = (1/2) * const1 * λ
            # add penalty term 
            for j in 1:ngroups(vcm)
                if √(vcm.Σ[j] + vcm.Σint[j]) < const1
                    pen += λ * √(vcm.Σ[j] + vcm.Σint[j]) - 
                        (vcm.Σ[j] + vcm.Σint[j]) / (2*penfun.γ)
                else
                    pen += const2
                end 
            end 
            obj += pen 
        end 
    end 

    return obj 
end 

"""
    nullprojection(Y, X, G)

Project `Y` to null space of `transpose(X)` and transform `G` accordingly.

# Input 
- `Y`: response vector/matrix to be transformed
- `X`: covariate matrix, response `Y` is projected onto the null space of transpose(X) 
- `G`: vector of genotype matrices, `(G[1],G[2],...,G[m])`

# Ouptut 
- `Ynew`: vectorized projected response vector/matrix 
- `Gnew`: vector of projected genotype matrices `(Gnew[1],Gnew[2],...,Gnew[m])`
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    Y    :: AbstractVecOrMat{T},
    X    :: AbstractVecOrMat{T},
    G    :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    if isempty(X)
        return vec(Y), G, X
    else
        # basis of nullspace of transpose(X), `N(X')`
        # Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
        # transpose!(Xt, X)
        B = nullspace(X')

        # dimension of null space 
        s = size(B, 2) 

        # projected response vector 
        Ynew = Matrix{T}(undef, s, size(Y, 2))
        BLAS.gemm!('T', 'N', one(T), B, Y, zero(T), Ynew)

        # no. of variance components subject to selection 
        m = length(G) 

        # transformed covariance matrices 
        Gnew = [Matrix{T}(undef, s, size(G[i], 2)) for i in 1:m]
        for i in 1:m
            BLAS.gemm!('T', 'N', one(T), B, G[i], zero(T), Gnew[i])
        end 

        # output 
        return vec(Ynew), Gnew, B 
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
    vcm :: Union{VCModel, VCintModel}
    ) 

    # quick return if there is no mean parameters
    isempty(vcm.β) && return vcm.β

    # 
    Ωchol = cholesky(Symmetric(vcm.Ωest))
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


"""
    formΩ!(Ω, Σ, G)

Overwrite `Ω` with `∑ Σi ⊗ GiGi'`

# Input 

# Output 
"""
function formΩ!(
    Ω :: AbstractMatrix{T}, 
    Σ :: AbstractVector{Matrix{T}}, 
    G :: AbstractVector{Matrix{T}}, 
    L :: AbstractMatrix{T} = Matrix{T}(undef, size(Σ[1], 1), size(Σ[1], 1))
    ) where {T <: Real}

    n = size(G[1], 1)
    d = size(Σ[1], 1) 
    Ω .= kron(Σ[end], I(n))
    tmp = Matrix{T}(undef, size(L, 1) * n, d)
    for i in 1:length(G)
        #println("Symmetric(Σ[i])=", Symmetric(Σ[i]))
        Σchol = cholesky(Symmetric(Σ[i]), Val(true), check=false)
        #dump(Σchol)
        #L[:] = Σchol.L[inv(Permutation(Σchol.p)).data, 1:Σchol.rank]
        L[:] = Σchol.L[inv(Permutation(Σchol.p)).data, 1:d]
        for j in 1:size(G[i], 2)
            fill!(tmp, 0)
            kronaxpy!(L, view(G[i], :, j), tmp)
            BLAS.syrk!('U', 'N', one(T), tmp, one(T), Ω)
        end
    end 
end 