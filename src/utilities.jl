"""
    nullprojection(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate matrix, response `y` is projected onto the null space of transpose(X) 
- `V`: vector of covariance matrices to be transformed, `(V[1],V[2],...,V[m],I)`
    note that V[end] should be identity matrix

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
    betaestimate(y, X, V, σ2)

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
function betaestimate( 
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

    β = betaestimate(y, X, Ω)

    return β

end 
"""
    betaestimate(y, X, Ω)

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
function betaestimate( 
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
    plotsolpath(solpath, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of solution path at varying λ values. First, calculate rank of each variance 
component based on the order in which it enters solution path. 
Use backend such as `gr()`.

# Input
- `solpath`: solution path (in numeric matrix) to be plotted, each column should 
        represent variance components at specific λ 
        as in output from `vcselect`, `vcselectpath`
- `λpath`: vector of tuning parameter λ values 

# Keyword 
- `title`: title of the figure, default is "Solution Path"
- `xlab`: x-axis label, default is minimum of λpath
- `ylab`: y-axis label, default is maximum of λpath
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6
- `linewidth`: line width, default is 0.7

# Output 
- plot of solution path 
"""
function plotsolpath(
    solpath   :: AbstractMatrix{T},
    λpath     :: AbstractVector{T};
    title     :: AbstractString = "Solution Path",
    xlab      :: AbstractString = "\\lambda",
    xmin      :: AbstractFloat = minimum(λpath),
    xmax      :: AbstractFloat = maximum(λpath),
    ylab      :: AbstractString = "\\sigma^2",
    linewidth :: AbstractFloat = 0.7,
    tol       :: Float64=1e-6
) where {T <: Real}

    # size of solution path 
    novarcomp, nlambda = size(solpath)

    # initialize array for ranking 
    ranking = Int[]

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
    tmp = findall(x -> x > tol, view(solpath, 1:(novarcomp-1), col))
    for j in tmp 
        if !(j in ranking)
        push!(ranking, j)
        end
    end
    end 
    rest = setdiff(1:(novarcomp-1), ranking)
    rest = [rest; novarcomp]

    # transpose solpath s.t. each row is estimates at particular lambda
    tr_solpath = solpath'

    # label in the permuted order 
    legendlabel = "\\sigma^{2}[$(ranking[1])]"
    for i in ranking[2:end]
        legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
    end
    for i in 1:length(rest)
        legendlabel = hcat(legendlabel, "")
    end 

    # plot permuted solution path (decreasing order)
    plot(λpath, tr_solpath[:, [ranking; rest]], label=legendlabel, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle="ranking")
    title!(title)     

end 