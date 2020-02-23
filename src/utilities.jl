"""
<<<<<<< HEAD
    nullprojection(y, X, V)

Project `y` to null space of `transpose(X)` and transform `V` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate vector or matrix, response `y` is projected onto the null space of transpose(X) 
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
    X    :: AbstractVecOrMat{T},
    V    :: AbstractVector{Matrix{T}};
    covariance :: Bool = true 
    ) where {T <: Real}

    # basis of nullspace of transpose(X), `N(X')`
    Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
    transpose!(Xt, X)
    B = nullspace(Xt)

    # projected response vector 
    ynew = B' * y 

    # dimension of null space 
    s = size(B, 2) 

    # no. of variance components subject to selection 
    m = length(V) - 1
    
    # initialize array 
    Vnew = Vector{Matrix{Float64}}(undef, m + 1)
    # 
    if covariance 
        Vnew[end] = Matrix{eltype(B)}(I, s, s) ./ √s
        tmp = zeros(size(X, 1), s)
        for i in 1:m
            mul!(tmp, V[i], B)
            Vnew[i] = BLAS.gemm('T', 'N', B, tmp)
            # divide by its frobenius norm  
            Vnew[i] ./= norm(Vnew[i])
        end 
    else 
        for i in eachindex(Vnew)
            Vnew[i] = BLAS.gemm('T', 'N', B, V[i])
        end 
    end 

    # output 
    return ynew, Vnew, B 

end 

"""
    nullprojection(y, X, V1, V2)

Project `y` to null space of `transpose(X)` and transform `V1` and `V2` accordingly.

# Input 
- `y`: response vector to be transformed. 
- `X`: covariate vector or matrix, response `y` is projected onto the null space of transpose(X) 
- `V1`: vector of covariance matrices, (V1[1],V1[2],...,V1[m],I/√n)
    note that each V[i] needs to have frobenius norm 1, and that V[end] should be 
    identity matrix divided by √n   
- `V2`: vector of covariance matrices, (V2[1],V2[2],...,V2[m],I/√n)
    note that each V[i] needs to have frobenius norm 1, and that V[end] should be 
    identity matrix divided by √n 

# Ouptut 
- `ynew`: projected response vector
- `V1new`: projected vector of covariance matrices, 
    frobenius norm of `V[i]` equals to 1 for all `i`
- `V2new`: projected vector of covariance matrices, 
    frobenius norm of `V[i]` equals to 1 for all `i`
- `B`: matrix whose columns are basis vectors of the null space of transpose(X) 
"""
function nullprojection(
    y  :: AbstractVector{T},
    X  :: AbstractVecOrMat{T},
    V1 :: AbstractVector{Matrix{T}},
    V2 :: AbstractVector{Matrix{T}}
    ) where {T <: Real}

    # basis of nullspace of transpose(X), `N(X')`
    Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
    transpose!(Xt, X)
    B = nullspace(Xt)

    # projected response vector 
    ynew = B' * y 

    # dimension of null space 
    s = size(B, 2) 

    # no. of variance components subject to selection 
    nvarcomps = length(V1) 

    # transformed covariance matrices 
    V1new = Vector{Matrix{Float64}}(undef, length(V1))
    V2new = Vector{Matrix{Float64}}(undef, length(V2))
    V1new[end] = Matrix{eltype(B)}(I, s, s) ./ √s
    tmp = zeros(size(X, 1), s)
    for i in 1:(nvarcomps - 1)
        mul!(tmp, V1[i], B)
        V1new[i] = BLAS.gemm('T', 'N', B, tmp)
        mul!(tmp, V2[i], B)
        V2new[i] = BLAS.gemm('T', 'N', B, tmp)
        # divide by its frobenius norm  
        V1new[i] ./= norm(V1new[i])
        V2new[i] ./= norm(V2new[i])
    end 

    # output 
    return ynew, V1new, V2new, B 
=======
    rankvarcomps(Σpath; tol=1e-8)

# Input
- `Σpath`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect!`, `vcselectpath!`
- `resvarcomp`: logical flag indicating the presence of residual variance component 
    in `Σpath`. If `true`, last row of `Σpath` is ignored. Default is `true`

# Keyword Argument 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-8 

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    Σpath      :: AbstractMatrix{T};
    tol        :: Float64 = 1e-8,
    resvarcomp :: Bool = true
    ) where {T <: Real}

    # size of solution path 
    novarcomps, nlambda = size(Σpath)

    # check if residual variance component is present in solution path 
    if resvarcomp 
        m = novarcomps - 1 
    else 
        m = novarcomps 
    end 

    # initialize array for ranking 
    ranking = Int[]
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        idx = findall(x -> x > tol, view(Σpath, 1:m, col))
        sortedidx = sortperm(Σpath[idx], rev=true)
        for j in idx[sortedidx] 
            if !(j in ranking)
                push!(ranking, j)
            end
        end
    end 
    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:m, ranking)
    if resvarcomp
        rest = [rest; novarcomps]
    end 
    
    return ranking, rest 
end 

"""
    rankvarcomps(Σpath; tol=1e-8)

# Input
<<<<<<< HEAD
- `y`: response vector
- `X`: covariate vector or matrix 
- `V`: vector of covariance matrices, (V[1],V[2],...,V[m],I)
        note that V[end] should be identity matrix
- `σ2`: REML estimate of variance components 
=======
- `Σpath`: solution path (matrix of matrices), each column should 
    contain estimated variance components at specific λ 
    as in output from `vcselectpath!`
- `resvarcomp`: logical flag indicating the presence of residual variance component 
    in `Σpath`. If `true`, last row of `Σpath` is ignored. Default is `true`

# Keyword Argument 
- `tol`: a variance component whose `p`-norm is less than `tol` is considered zero, 
    default is 1e-8 
- `p`: `p` for determining which norm to use, default is 2. See [`norm(A, p::Real=2)`](@ref)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of variance components that are estimated to be zero at all λ > 0
"""
<<<<<<< HEAD
function betaestimate( 
    y   :: AbstractVector{T},
    X   :: AbstractVecOrMat{T},
    V   :: AbstractVector{Matrix{T}},
    σ2  :: AbstractVector{T}
=======
function rankvarcomps(
    Σpath      :: AbstractMatrix{Matrix{T}};
    tol        :: Float64 = 1e-8,
    p          :: Real = 2,
    resvarcomp :: Bool = true
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
    ) where {T <: Real}

    # size of solution path 
    novarcomps, nlambda = size(Σpath)

    # check if residual variance component is present in solution path 
    if resvarcomp 
        m = novarcomps - 1 
    else 
        m = novarcomps 
    end

    # initialize array for ranking 
    ranking = Int[]

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        idx = findall(x -> norm(x, p) > tol, view(Σpath, 1:m, col))
        sortedidx = sortperm(norm.(Σpath[idx]), rev=true)
        for j in idx[sortedidx] 
            if !(j in ranking)
                push!(ranking, j)
            end
        end
    end 
    
    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:m, ranking)
    if resvarcomp 
        rest = [rest; novarcomps]
    end 

    return ranking, rest 
end 
<<<<<<< HEAD

"""

"""
function betaestimate( 
    y   :: AbstractVector{T},
    X   :: AbstractVecOrMat{T},
    V   :: AbstractVector{Matrix{T}},
    Vint :: AbstractVector{Matrix{T}},
    σ2  :: AbstractVector{T},
    σ2int  :: AbstractVector{T}
    ) where {T <: Real}

    # update Ω with estimated variance components 
    Ω = zeros(T, size(V[1]))
    for i in eachindex(σ2int)
        if iszero(σ2[i]) && iszero(σ2int[i])
            continue 
        else 
            axpy!(σ2[i], V[i], Ω) 
            axpy!(σ2int[i], Vint[i], Ω) 
        end 
    end 
    axpy!(σ2[end], V[end], Ω)

    β = betaestimate(y, X, Ω)

    return β

end 
"""
    betaestimate(y, X, Ω)
=======
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

"""
    rankvarcomps(Σpath, Σpath2; tol=1e-6, resvarcomp=true, resvarcomp2=true)

Obtain rank of variance components from a paired solution paths, e.g. `Σpath` and `Σpath2`.   
Ranks are calculated using norm of paired variance components. 
# Input
<<<<<<< HEAD
- `y`: response vector
- `X`: covariate vector or matrix 
- `Ω`: overall covariance matrix constructed using REML estimate of variance components or
    cholesky factorization of the overall covariance matrix 

=======
- `Σpath`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `Σpath2`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
# Keyword 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6 
- `resvarcomp`: logical flag indicating there is residual variance component in `Σpath`, 
    default is true. If true, the last variance component is not included in ranks
- `resvarcomp2`: logical flag indicating there is residual variance component in `Σpath2`,
   default is true. If true, the last variance component is not included in ranks
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
# Output 
- `ranks`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
<<<<<<< HEAD
function betaestimate( 
    y   :: AbstractVector{T},
    X   :: AbstractVecOrMat{T},
    Ω   :: Union{AbstractMatrix{T}, Cholesky}
=======
function rankvarcomps(
    Σpath      :: AbstractMatrix{T},
    Σpath2     :: AbstractMatrix{T};
    tol         :: Float64 = 1e-6,
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
    ) where {T <: Real}

    @assert size(Σpath, 2) == size(Σpath2, 2) "both solution path should have the same number of tuning parameters!\n"

    # size of solution path 
    nvarcomps, nlambda = size(Σpath)
    nvarcomps2, nlambda2 = size(Σpath2)

    if resvarcomp 
        m = nvarcomps - 1 
    else 
        m = nvarcomps 
    end 
<<<<<<< HEAD
  
    # estimate fixed effects: pinv(X'*Ωinv*X)(X'*Ωinv*y)
    if typeof(X) <: Vector 
        β = (X' * (Ωchol \ y)) / (X' * (Ωchol \ X))
    else 
        XtΩinvX = BLAS.gemm('T', 'N', X, Ωchol \ X)
        β = BLAS.gemv('T', X, Ωchol \ y) # overwriting Ωinv with X'*Ωinv
        β = pinv(XtΩinvX) * β
    end 
=======
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

    if resvarcomp2 
        m2 = nvarcomps2 - 1 
    else 
        m2 = nvarcomps2
    end 

    @assert m == m2 "solution paths need to have the same number of variance components!\n"

    # initialize array for ranks
    ranks = Int[]
    normpath = similar(Σpath)

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        bothpath = [view(Σpath, 1:m, col) view(Σpath2, 1:m2, col)]
        normpath[1:m, col] = mapslices(norm, bothpath; dims=2) #mapslices(x -> norm(x, p), bothpath; dims=2)
        idx = findall(x -> x > tol, view(normpath, 1:m, col))
        sortedidx = sortperm(normpath[idx, col], rev=true)
        for j in idx[sortedidx]
            if !(j in ranks)
                push!(ranks, j)
            end
        end
    end 
    normpath[1:m, 1] = mapslices(norm, [view(Σpath, 1:m, 1) view(Σpath2, 1:m2, 1)]; dims=2)
    normpath[end, :] .= Σpath[end, :]

    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:m, ranks)
    if resvarcomp 
        rest = [rest; nvarcomps]
    end 

    return ranks, rest, normpath 
end 

"""
    plotsolpath(Σpath, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of solution path at varying λ values. Use backend such as `gr()`.

# Input
- `Σpath`: solution path (in numeric matrix) to be plotted, each column should 
        represent variance components at specific λ as in output from `vcselectpath!`. 
	The last row of `Σpath` should be for residual variance component. Otherwise set `resvarcomp=false` to indicate the absence. 
- `λpath`: vector of tuning parameter λ values.

# Keyword 
- `title`: title of the figure. Default is "Solution Path".
- `xlab`: x-axis label. Default is "lambda".
- `ylab`: y-axis label. Default is "sigma_i^2".
- `xmin`: lower limit for x-axis. default is minimum of `λpath`.
- `xmax`: upper limit for x-axis. default is maximum of `λpath`.
- `linewidth`: line width. Default is 1.0.
- `nranks`: no. of ranks to displayed on legend. Default is total number of variance components.
- `legend`: logical flag for including legend. Default is true.
- `legendout`: logical flag for moving the legend outside the plot. Default is true. 
- `legendtitle`: legend title. Default is "Ranking". 
- `resvarcomp`: logical flag for indicating residual variance component in `Σpath`. 	   
      Default is true. 

# Output 
- plot of solution path. 
"""
function plotsolpath(
    Σpath     :: AbstractMatrix{T},
    λpath      :: AbstractVector{T};
    title      :: AbstractString = "Solution Path",
    xlab       :: AbstractString = "\\lambda",
    ylab       :: AbstractString = "\\sigma^2",
    xmin       :: AbstractFloat = minimum(λpath),
    xmax       :: AbstractFloat = maximum(λpath),
    linewidth  :: AbstractFloat = 1.0,
    nranks     :: Int = size(Σpath, 1),
    legend     :: Bool = true,
    legendout  :: Bool = true, 
    legendtitle :: AbstractString = "Ranking",
    resvarcomp :: Bool = true
    ) where {T <: Real}

    # size of solution path 
    novarcomps, nlambda = size(Σpath)

    # get ranking of variance components
    ranking, rest = rankvarcomps(Σpath; resvarcomp=resvarcomp)

    # transpose solpath s.t. each row is estimates at particular lambda
    tr_Σpath = Σpath'

   if legend && nranks > 0 
        legendlabel = "\\sigma^{2}[$(ranking[1])]"
        if nranks == novarcomps # display all non-zero variance components 
            
            for i in ranking[2:end]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end
            nranks = length(ranking)
           
        elseif nranks > 1 # display the first non-zero variance component to enter the path  
            for i in ranking[2:nranks]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end

        end 

        for i in 1:(novarcomps - nranks)
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)

        if !legendout
            plot(λpath, tr_Σpath[:, [ranking; rest]], label=legendlabel, 
            xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle=legendtitle)
            title!(title) 
        else 
            pt1 = plot(λpath, tr_Σpath[:, [ranking; rest]],  legend=false,
                xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, 
                legendtitle=legendtitle)
            title!(title)
            pt2 = plot(λpath, tr_Σpath[:, [ranking; rest]], label=legendlabel, grid=false, 
                        showaxis=false, xlims=(20,3), legendtitle=legendtitle) 
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
        plot(λpath, tr_Σpath[:, [ranking; rest]], legend=false, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 
   
end 
<<<<<<< HEAD
 
"""
    rankvarcomps(σ2path; tol=1e-6, resvarcomp=true)

Obtain rank of variance components from a solution path

# Input
- `σ2path`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`

# Keyword 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6 
- `resvarcomp`: logical flag indicating there is residual variance component, default is true
    if true, the last variance component is not included in calculating ranks

# Output 
- `ranks`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    σ2path     :: AbstractMatrix{T};
    tol        :: Float64 = 1e-6,
    resvarcomp :: Bool = true  
    ) where {T <: Real}

    # size of solution path 
    nvarcomps, nlambda = size(σ2path)

    if resvarcomp 
        m = nvarcomps - 1 
    else 
        m = nvarcomps 
    end 

    # initialize array for ranks 
    ranks = Int[]

    # fill in the array 
    for col in nlambda:-1:2
        idx = findall(x -> x > tol, view(σ2path, 1:m, col))
        sortedidx = sortperm(σ2path[idx], rev=true)
        for j in idx[sortedidx]
            if !(j in ranks)
                push!(ranks, j)
            end
        end
    end 

    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:m, ranks)
    if resvarcomp 
        rest = [rest; nvarcomps]
    end 

    return ranks, rest 
end 

"""
    rankvarcomps(σ2path, σ2path2; tol=1e-6, resvarcomp=true, resvarcomp2=true)

Obtain rank of variance components from a paired solution paths, e.g. `σ2path` and `σ2intpath`.   
Ranks are calculated using norm of paired variance components. 

# Input
- `σ2path`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `σ2path2`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`

# Keyword 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6 
- `resvarcomp`: logical flag indicating there is residual variance component in `σ2path`, 
    default is true. If true, the last variance component is not included in ranks
- `resvarcomp2`: logical flag indicating there is residual variance component in `σ2path2`,
   default is true. If true, the last variance component is not included in ranks

# Output 
- `ranks`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    σ2path      :: AbstractMatrix{T},
    σ2path2     :: AbstractMatrix{T};
    tol         :: Float64 = 1e-6,
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
=======

"""
    plotsolpath(Σpath, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of solution path at varying λ values. Use backend such as `gr()`.

# Input
- `Σpath`: solution path (in numeric matrix) to be plotted, each column should 
        represent variance components at specific λ as in output from `vcselectpath!`. 
        The last row of `Σpath` should be for residual variance component. 
        Otherwise set `resvarcomp=false` to indicate the absence. 
- `λpath`: vector of tuning parameter λ values.

# Keyword 
- `title`: title of the figure. Default is "Solution Path".
- `xlab`: x-axis label. Default is "lambda".
- `ylab`: y-axis label. Default is "sigma_i^2".
- `xmin`: lower limit for x-axis. default is minimum of `λpath`.
- `xmax`: upper limit for x-axis. default is maximum of `λpath`.
- `linewidth`: line width. Default is 1.0.
- `nranks`: no. of ranks to displayed on legend. Default is total number of variance components.
- `legend`: logical flag for including legend. Default is true.
- `legendout`: logical flag for moving the legend outside the plot. Default is true. 
- `legendtitle`: legend title. Default is "Ranking". 
- `resvarcomp`: logical flag for indicating residual variance component in `Σpath`. 	   
      Default is true. 

# Output 
- plot of solution path.
"""
function plotsolpath(
    Σpath      :: AbstractMatrix{Matrix{T}},
    λpath      :: AbstractVector{T};
    title      :: AbstractString = "Solution Path",
    xlab       :: AbstractString = "\\lambda",
    ylab       :: AbstractString = "||\\Sigma||_2",
    xmin       :: AbstractFloat = minimum(λpath),
    xmax       :: AbstractFloat = maximum(λpath),
    linewidth  :: AbstractFloat = 1.0,
    nranks     :: Int = size(Σpath, 1),
    legend     :: Bool = true,
    legendout  :: Bool = true, 
    legendtitle :: AbstractString = "Ranking",
    resvarcomp :: Bool = true
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
    ) where {T <: Real}

    @assert size(σ2path, 2) == size(σ2path2, 2) "both solution path should have the same number of tuning parameters!\n"

    # size of solution path 
<<<<<<< HEAD
    nvarcomps, nlambda = size(σ2path)
    nvarcomps2, nlambda2 = size(σ2path2)

    if resvarcomp 
        m = nvarcomps - 1 
    else 
        m = nvarcomps 
    end 

    if resvarcomp2 
        m2 = nvarcomps2 - 1 
    else 
        m2 = nvarcomps2
    end 

    @assert m == m2 "solution paths need to have the same number of variance components!\n"

    # initialize array for ranks
    ranks = Int[]
    normpath = similar(σ2path)

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        bothpath = [view(σ2path, 1:m, col) view(σ2path2, 1:m2, col)]
        normpath[1:m, col] = mapslices(norm, bothpath; dims=2)
        idx = findall(x -> x > tol, view(normpath, 1:m, col))
        sortedidx = sortperm(normpath[idx, col], rev=true)
        for j in idx[sortedidx]
            if !(j in ranks)
                push!(ranks, j)
=======
    novarcomps, nlambda = size(Σpath)

    # get ranking of variance components
    ranking, rest = rankvarcomps(Σpath; resvarcomp=resvarcomp)

    # transpose solpath s.t. each row is estimates at particular lambda
    tr_Σpath = norm.(Σpath)'

   if legend && nranks > 0 
        legendlabel = "\\Sigma[$(ranking[1])]"
        if nranks == novarcomps # display all non-zero variance components 
            
            for i in ranking[2:end]
                legendlabel = hcat(legendlabel, "\\Sigma[$i]")
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
            end
            nranks = length(ranking)
           
        elseif nranks > 1 # display the first non-zero variance component to enter the path  
            for i in ranking[2:nranks]
                legendlabel = hcat(legendlabel, "\\Sigma[$i]")
            end

        end 

        for i in 1:(novarcomps - nranks)
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)

        if !legendout
            plot(λpath, tr_Σpath[:, [ranking; rest]], label=legendlabel, 
            xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle=legendtitle)
            title!(title) 
        else 
            pt1 = plot(λpath, tr_Σpath[:, [ranking; rest]],  legend=false,
                xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, 
                legendtitle=legendtitle)
            title!(title)
            pt2 = plot(λpath, tr_Σpath[:, [ranking; rest]], label=legendlabel, grid=false, 
                        showaxis=false, xlims=(20,3), legendtitle=legendtitle) 
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
        plot(λpath, tr_Σpath[:, [ranking; rest]], legend=false, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 
<<<<<<< HEAD
    normpath[1:m, 1] = mapslices(norm, [view(σ2path, 1:m, 1) view(σ2path2, 1:m2, 1)]; dims=2)
    normpath[end, :] .= σ2path[end, :]

    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:m, ranks)
    if resvarcomp 
        rest = [rest; nvarcomps]
    end 

    return ranks, rest, normpath 
=======

>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
end 

"""
    plotsolpath(Σpath, Σpath2, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)
Output plot of a paired solution path at varying λ values. Use backend such as `gr()`.

# Input
- `Σpath`: solution path (in numeric matrix) to be plotted, each column should 
    represent variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `Σpath2`: solution path (in numeric matrix) to be plotted, each column should 
    represent variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `λpath`: vector of tuning parameter λ values 

# Keyword 
<<<<<<< HEAD
- `title`: title of the figure, default is "Solution Path"
- `xlab`: x-axis label, default is minimum of λpath
- `ylab`: y-axis label, default is maximum of λpath
- `nranks`: no. of ranks to display on legend, default is total number of variance components
- `linewidth`: line width, default is 1.0
- `legend`: indicator to include legend or not, default is true 
- `legendout`: indicator to move legend outside the plot, default is true 
=======
- `title`: title of the figure. Default is "Solution Path".
- `xlab`: x-axis label. Default is "lambda".
- `ylab`: y-axis label. Default is "sigma_i^2".
- `xmin`: lower limit for x-axis. default is minimum of `λpath`.
- `xmax`: upper limit for x-axis. default is maximum of `λpath`.
- `linewidth`: line width. Default is 1.0.
- `nranks`: no. of ranks to displayed on legend. Default is total number of variance components.
- `legend`: logical flag for including legend. Default is true.
- `legendout`: logical flag for moving the legend outside the plot. Default is true. 
- `legendtitle`: legend title. Default is "Ranking". 
- `resvarcomp`: logical flag for indicating residual variance component in `Σpath`. 	   
      Default is true. 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

# Output 
- plot of solution path 
"""
function plotsolpath(
<<<<<<< HEAD
    σ2path    :: AbstractMatrix{T},
    λpath     :: AbstractVector{T};
    title     :: AbstractString = "Solution Path",
    xlab      :: AbstractString = "\$\\lambda\$",
    xmin      :: AbstractFloat = minimum(λpath),
    xmax      :: AbstractFloat = maximum(λpath),
    ylab      :: AbstractString = "\$\\sigma_i^2\$",
    nranks    :: Int = size(σ2path, 1),
    linewidth :: AbstractFloat = 1.0, 
    legend    :: Bool = true,
    legendout :: Bool = false
=======
    Σpath       :: AbstractMatrix{T},
    Σpath2      :: AbstractMatrix{T},
    λpath       :: AbstractVector{T};
    title       :: AbstractString = "Solution Path",
    xlab        :: AbstractString = "\$\\lambda\$",
    xmin        :: AbstractFloat = minimum(λpath),
    xmax        :: AbstractFloat = maximum(λpath),
    ylab        :: AbstractString = "\$||(\\sigma_{i1}^2, \\sigma_{i2}^2)||_2\$",
    nranks      :: Int = size(Σpath, 1),
    linewidth   :: AbstractFloat = 1.0, 
    legend      :: Bool = true,
    legendout   :: Bool = false,
    legendtitle :: AbstractString = "Ranking",
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
) where {T <: Real}

    # error handling 
    @assert size(Σpath, 2) == size(Σpath2, 2) "both solution paths must have the same number of lambdas!\n"

    # size of solution path 
<<<<<<< HEAD
    nvarcomps, nlambda = size(σ2path)
=======
    m, nlambda = size(Σpath2)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

    # get ranking of variance components
    ranking, rest, normpath = rankvarcomps(Σpath, Σpath2; 
            resvarcomp=resvarcomp, resvarcomp2=resvarcomp2)

    # transpose solpath s.t. each row is estimates at particular lambda
    trnormpath = normpath'

    if legend && nranks > 0 
        legendlabel = "\\sigma^{2}[$(ranking[1])]"
        if nranks == m+1 # display all non-zero variance components 
            
            for i in ranking[2:end]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end
            nranks = length(ranking)
           
        elseif nranks > 1 # display the first non-zero variance component to enter the path  
            for i in ranking[2:nranks]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end

<<<<<<< HEAD
    if legend && nranks > 0 
        legendlabel = "\\sigma^{2}[$(ranking[1])]"
        if nranks == nvarcomps # display all non-zero variance components 
            
            for i in ranking[2:end]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end
            nranks = length(ranking)
           
        elseif nranks > 1 # display the first non-zero variance component to enter the path  
            for i in ranking[2:nranks]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end

        end 

        for i in 1:(nvarcomps - nranks)
=======
        end 

        for i in 1:(m + 1 - nranks)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)
<<<<<<< HEAD

        if !legendout
            plot(λpath, tr_σ2path[:, [ranking; rest]], label=legendlabel, 
            xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle="ranking")
            title!(title) 
        else 
            pt1 = plot(λpath, tr_σ2path[:, [ranking; rest]],  legend=false,
                xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, 
                legendtitle="ranking")
            title!(title)
            pt2 = plot(λpath, tr_σ2path[:, [ranking; rest]], label=legendlabel, grid=false, 
                        showaxis=false, xlims=(20,3), legendtitle="ranking") 
=======
        if !legendout
            plot(λpath, trnormpath[:, [ranking; rest]], label=legendlabel, 
            xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle=legendtitle)
            title!(title) 
        else 
            pt1 = plot(λpath, trnormpath[:, [ranking; rest]],  legend=false,
                xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, 
                legendtitle=legendtitle)
            title!(title)
            pt2 = plot(λpath, trnormpath[:, [ranking; rest]], label=legendlabel, grid=false, 
                        showaxis=false, xlims=(20,3), legendtitle=legendtitle) 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
<<<<<<< HEAD
        plot(λpath, tr_σ2path[:, [ranking; rest]], legend=false, 
=======
        plot(λpath, trnormpath[:, [ranking; rest]], legend=false, 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 

end 

<<<<<<< HEAD
"""
    plotsolpath(σ2path, σ2intpath, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of a paired solution path at varying λ values. Use backend such as `gr()`.

# Input
- `σ2path`: solution path (in numeric matrix) to be plotted, each column should 
    represent variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `σ2intpath`: solution path (in numeric matrix) to be plotted, each column should 
    represent variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`
- `λpath`: vector of tuning parameter λ values 

# Keyword 
- `title`: title of the figure, default is "Solution Path"
- `xlab`: x-axis label, default is minimum of λpath
- `ylab`: y-axis label, default is maximum of λpath
- `nranks`: no. of ranks to display on legend, default is total number of variance components
- `linewidth`: line width, default is 1.0
- `legend`: indicator to include legend or not, default is true 
- `legendout`: indicator to move legend outside the plot, default is true 

# Output 
- plot of solution path 
"""
function plotsolpath(
    σ2path      :: AbstractMatrix{T},
    σ2intpath   :: AbstractMatrix{T},
    λpath       :: AbstractVector{T};
    title       :: AbstractString = "Solution Path",
    xlab        :: AbstractString = "\$\\lambda\$",
    xmin        :: AbstractFloat = minimum(λpath),
    xmax        :: AbstractFloat = maximum(λpath),
    ylab        :: AbstractString = "\$||(\\sigma_{i1}^2, \\sigma_{i2}^2)||_2\$",
    nranks      :: Int = size(σ2path, 1),
    linewidth   :: AbstractFloat = 1.0, 
    legend      :: Bool = true,
    legendout   :: Bool = false,
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
) where {T <: Real}

    # error handling 
    @assert size(σ2path, 2) == size(σ2intpath, 2) "both solution paths must have the same number of lambdas!\n"

    # size of solution path 
    m, nlambda = size(σ2intpath)

    # get ranking of variance components
    ranking, rest, normpath = rankvarcomps(σ2path, σ2intpath; 
            resvarcomp=resvarcomp, resvarcomp2=resvarcomp2)

    # transpose solpath s.t. each row is estimates at particular lambda
    trnormpath = normpath'

    if legend && nranks > 0 
        legendlabel = "\\sigma^{2}[$(ranking[1])]"
        if nranks == m+1 # display all non-zero variance components 
            
            for i in ranking[2:end]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end
            nranks = length(ranking)
           
        elseif nranks > 1 # display the first non-zero variance component to enter the path  
            for i in ranking[2:nranks]
                legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
            end

        end 

        for i in 1:(m + 1 - nranks)
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)
        if !legendout
            plot(λpath, trnormpath[:, [ranking; rest]], label=legendlabel, 
            xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle="ranking")
            title!(title) 
        else 
            pt1 = plot(λpath, trnormpath[:, [ranking; rest]],  legend=false,
                xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, 
                legendtitle="ranking")
            title!(title)
            pt2 = plot(λpath, trnormpath[:, [ranking; rest]], label=legendlabel, grid=false, 
                        showaxis=false, xlims=(20,3), legendtitle="ranking") 
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
        plot(λpath, trnormpath[:, [ranking; rest]], legend=false, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 
=======

>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

