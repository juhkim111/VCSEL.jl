"""
    projectToNullSpace(y, X, V; tol=1e-6)

Project `y` to null space of `X` and transform `V` accordingly.

# Input 
- `y::Vector{Float64}`: response vector to be transformed. 
- `X::Matrix{Float64}`: covariate matrix whose null space `y` is projected to.
- `V::Vector{Matrix{Float64}}`: vector of covariance matrices to be transformed. 
- `tol::Float64`: if any diagonal entries  `Q` matrix in QR decomposition has its absolute
    value samller or equal to `tol`, then it is considered 0. That column is considered a
    basis vector of null space of `I-X(X'X)^{-1}X'`. Default is 1e-6. 

# Ouptut 
- `ynew`: projected response vector.
- `Vnew`: projected vector of covariance matrices. 

"""
function projectToNullSpace(
    y    :: Vector{T},
    X    :: Matrix{T},
    V    :: Vector{Matrix{T}};
    tol  :: T = 1e-6
    ) where {T <: Float64}

    # number of groups 
    m = length(V) - 1

    ## REML: find B s.t. columns of B span the null space of X' and B'B = I
    # pivoted QR factorization of I-X(X'X)^{-1}X'
    QRfact = qr(I - X * inv(cholesky(X' * X)) * X', Val(true))
    # extract orthonormal basis of C(I-P)
    B = QRfact.Q[:, abs.(diag(QRfact.R)) .> tol] 
    # REML transformed response vector 
    ynew = B' * y
    # REML transformed covariance matrices 
    Vnew  = Array{Matrix{Float64}}(undef, m + 1)
    for i in 1:(m + 1)
        Vnew[i] = B' * V[i] * B  
    end  

    # make sure frobenius norm of Vi equals to 1 
    for i in 1:(m + 1)
        if norm(Vnew[i]) != 1
            Vnew[i] ./= norm(Vnew[i])
        end 
    end 
    
    # output 
    return ynew, Vnew 

end 