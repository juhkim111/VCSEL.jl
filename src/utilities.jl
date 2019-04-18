"""
    projectontonull(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate matrix, response `y` is projected onto the null space of transpose(X) 
- `V`: vector of covariance matrices to be transformed, (V[1],V[2],...,V[m],I)
    note that V[end] should be identity matrix

# Ouptut 
- `ynew`: projected response vector
- `Vnew`: projected vector of covariance matrices
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 

"""
function projectontonull(
    y    :: AbstractVector{T},
    X    :: AbstractMatrix{T},
    V    :: AbstractVector{AbstractMatrix{T}}
    ) where {T <: Real}

    # basis of nullspace of transpose(X), `N(X')`
    Xt = similar(X')
    transpose!(Xt, X)
    B = nullspace(Xt)

    # projected response vector 
    ynew = B' * y 

    # transformed covariance matrices 
    Vnew = similar(V)
    for i in 1:(length(V) - 1) 
        mul!(Vnew[i], B' * V[i], B)
        # make sure frobenius norm equals to 1
        Vnew[i] ./= norm(Vnew[i])
    end 
    
    # output 
    return ynew, Vnew, B 

end 