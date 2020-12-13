export nullprojection, kronaxpy!, clamp_diagonal!, updateM!

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
    nullprojection(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector/matrix to be transformed
- `X`: covariate matrix, response `y` is projected onto the null space of transpose(X) 
- `G`: vector of genotype matrices, `(G[1],G[2],...,G[m])`
    note `V[end]` should be identity matrix or identity matrix divided by √n

# Ouptut 
- `ynew`: projected response vector/matrix
- `Vnew`: projected vector of covariance matrices
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    y    :: AbstractVecOrMat{T},
    X    :: AbstractVecOrMat{T},
    G    :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    if isempty(X)
        return y, G, X
    else
        # basis of nullspace of transpose(X), `N(X')`
        B = nullspace(X')

        # projected response vector 
        ynew = B' * y 

        # dimension of null space 
        s = size(B, 2) 

        # 
        m = length(G)

        # transformed covariance matrices 
        Gnew = Vector{Matrix{Float64}}(undef, m)
        for i in 1:m
            Gnew[i] = BLAS.gemm('T', 'N', B, G[i])
        end 

        # output 
        return ynew, Gnew, B 
    end 

end 


# """
#     updateβ!(vcm)

# Estimate fixed effects using REML estimate of variance components.
# Estimate of beta is 
#     `beta = pinv(X'*Ωinv*X)(X'*Ωinv*y)`
#     `beta = pinv()`
# where `Ω` being `∑ σ2[i] * V[i]` or `∑ Σ[i] ⊗ V[i]` where `σ2` or `Σ` are REML estimates.

# # Input
# - `vcm`: VCModel

# # Output 
# - `vcm.β`: updated fixed parameter estimates 
# """
# function updateβ!( 
#     vcm :: Union{VCModel}
#     ) 

#     # quick return if there is no mean parameters
#     isempty(vcm.β) && return vcm.β

#     # 
#     Ωchol = cholesky(Symmetric(vcm.Ωest))
#     # 
#     d = length(vcm)
#     p = size(vcm.Xobs, 2)
#     kron_I_X = kron(Matrix(I, d, d), vcm.Xobs)
#     XtΩinvX = BLAS.gemm('T', 'N', kron_I_X, Ωchol \ kron_I_X)
#     β = BLAS.gemv('T', kron_I_X, Ωchol \ vec(vcm.Yobs))
#     β = pinv(XtΩinvX) * β
#     if d == 1 
#         vcm.β .= β
#     else 
#         vcm.β .= reshape(β, p, d)
#     end 
#     vcm.β
# end

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
    mul!(res, Ω, M)

Calculates the matrix-matrix or matrix-vector product ``ΩM`` and stores the result in `res`,
overwriting the existing value of `res`. Note that `res` must not be aliased with either `Ω` or
`M`.
"""
function LinearAlgebra.mul!(
    res :: AbstractVecOrMat{T}, 
    Ω   :: MvcCovMatrix{T}, 
    M   :: AbstractVecOrMat{T}) where T <: Real
    n, d, m = size(Ω.G[1], 1), size(Ω.Σ[1], 1), length(Ω.G)
    Σ, G = Ω.Σ, Ω.G
    for col in 1:size(M, 2)
        Mcol = reshape(view(M, :, col), n, d)
        rcol = view(res, :, col)
        mul!(Ω.storage_nd, Mcol, Σ[end])
        copyto!(rcol, Ω.storage_nd)
        for i in 1:m
            q = size(G[i], 2)
            mul!(view(Ω.storage_qd_1, 1:q, :), transpose(G[i]), Mcol)
            mul!(view(Ω.storage_qd_2, 1:q, :), view(Ω.storage_qd_1, 1:q, :), Σ[i])
            mul!(Ω.storage_nd, G[i], view(Ω.storage_qd_2, 1:q, :))
            @inbounds @simd for j in 1:length(rcol)
                rcol[j] += Ω.storage_nd[j]
            end 
        end
    end
    res
end

""" 
    updateM!(M, )

Calculate `M = (I_d⊗1_n)'[(1_d1_d'⊗GG')⊙Ωinv](I_d⊗1_n)`. For multi-threading, start julia
with `julia --threads 4` ()

# Input 
- `M`: d x d matrix 
- `G`: n x q matrix  
- `rhs_nd_q`: nd x q matrix 
- `storage_nd_q`: initial guess of `Ωinv (I_d⊗G)`
- `Ω`: MvcCovMatrix
- `C`

# Output 
- `obj`: objective value 

"""
function updateM!(
    M     :: Matrix{T},
    G     :: Matrix{T},
    Ω     :: MvcCovMatrix{T},
    rhs_nd     :: Vector{T} = Vector{T}(undef, size(G, 1) * size(M, 1)),
    #rhs_nd_q     :: Matrix{T} = Matrix{T}(undef, size(G, 1) * size(M, 1), size(G, 2)), 
    storage_nd_q :: Matrix{T} = Matrix{T}(undef, size(G, 1) * size(M, 1), size(G, 2)),
    C     :: Matrix{T} = Matrix{T}(undef, size(G, 2), size(G, 2))
    ) where T <: Real 
    n, d, q = size(G, 1), size(M, 1), size(G, 2)
 
    # loop over each block 
    for j in 1:d 
        # obtain Ωinv * (I ⊗ G)
            # TRIED, but failed: get NaN
            # for k in 1:q
            #     for iter in 1:((j-1)*n)
            #         rhs_nd_q[iter, k] = zero(T)
            #     end
            #     for iter in 1:n
            #         rhs_nd_q[(j-1)*n + iter, k] = G[iter, k]
            #     end 
            #     for iter in (j * n + 1):(n*d)
            #         rhs_nd_q[iter, k] = zero(T)
            #     end
            # end
            # Threads.@threads for k in 1:q
            #     cg!(view(storage_nd_q, :, k), Ω, view(rhs_nd_q, :, k))
            # end

        # THIS WORKS: 96.610988 seconds (947.39 k allocations: 54.333 MiB, 0.01% gc time)
        for k in 1:q 
            for iter in 1:((j-1)*n)
                rhs_nd[iter] = zero(T)
            end
            for iter in 1:n
                rhs_nd[(j-1)*n + iter] = G[iter, k]
            end 
            for iter in (j * n + 1):(n*d)
                rhs_nd[iter] = zero(T)
            end
            copyto!(view(storage_nd_q, :, k), cg(Ω, rhs_nd))
        end


        #fill in Mi matrix 
        for Mrow in 1:d 
            #M[Mrow, j] = tr(G' * storage_nd_q[(((Mrow-1)* n)+1):(((Mrow -1) * n)+n), 1:q])
            M[Mrow, j] = tr(BLAS.gemm!('T', 'N', one(T), G, 
                        storage_nd_q[(((Mrow-1)* n)+1):(((Mrow -1) * n)+n), 1:q], zero(T), C))
        end
    end 

    return M 
end 

