"""
    projectToNullSpace(y, X, V; tol=1e-6)

Project `y` to null space of `X` and transform `V` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate matrix whose null space `y` is projected to.
- `V`: vector of covariance matrices to be transformed. 
- `tol`: if any diagonal entries  `Q` matrix in QR decomposition has its absolute
    value smaller or equal to `tol`, then it is considered 0. That column is considered a
    basis vector of null space of `I-X(X'X)^{-1}X'`. Default is 1e-6. 

# Ouptut 
- `ynew`: projected response vector.
- `Vnew`: projected vector of covariance matrices. 

"""
function projectToNullSpace(
    y    :: Vector{T},
    X    :: Matrix{T},
    V    :: Vector{Matrix{T}};
    tol  :: Float64 = 1e-6
    ) where {T <: Real}

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
    Vnew  = Array{Matrix{T}}(undef, m + 1)
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