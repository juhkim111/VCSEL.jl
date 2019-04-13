"""
    projectToNullSpace(yobs, Xobs, Vobs; tol=1e-6)

Project `yobs` to null space of `Xobs` and transform `Vobs` accordingly.

# Input 
- `yobs::Vector{Float64}`: response vector to be transformed. 
- `Xobs::Matrix{Float64}`: covariate matrix whose null space `yobs` is projected to.
- `Vobs::Vector{Matrix{Float64}}`: vector of covariance matrices to be transformed. 
- `tol::Float64`: if any diagonal entries  `Q` matrix in QR decomposition has its absolute
    value samller or equal to `tol`, then it is considered 0. That column is considered a
    basis vector of null space of `I-Xobs(Xobs'Xobs)^{-1}Xobs'`. Default is 1e-6. 

# Ouptut 
- `y`: projected response vector.
- `V`: projected vector of covariance matrices. 

"""
function projectToNullSpace(
    yobs    :: Vector{Float64},
    Xobs    :: Matrix{Float64},
    Vobs    :: Vector{Matrix{Float64}};
    tol     :: Float64 = 1e-6
)

    # number of groups 
    m = length(Vobs) - 1

    ## REML: find B s.t. columns of B span the null space of X' and B'B = I
    # pivoted QR factorization of I-X(X'X)^{-1}X'
    QRfact = qr(I - Xobs * inv(cholesky(Xobs' * Xobs)) * Xobs', Val(true))
    # extract orthonormal basis of C(I-P)
    B = QRfact.Q[:, abs.(diag(QRfact.R)) .> tol] 
    # REML transformed response vector 
    y = B' * yobs 
    # REML transformed covariance matrices 
    V  = Array{Matrix{Float64}}(undef, m + 1)
    for i in 1:(m + 1)
        V[i] = B' * Vobs[i] * B  
    end  

    # make sure frobenius norm of Vi equals to 1 
    for i in 1:(m + 1)
        if norm(V[i]) != 1
            V[i] ./= norm(V[i])
        end 
    end 
    
    # output 
    return y, V 

end 