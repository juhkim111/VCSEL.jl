export plotsolpath, checkfrobnorm!, rankvarcomps, matarray2mat

"""
    checkfrobnorm!(V)

Check if frobenius norm of Vi in V equals to 1. If not, divide by its norm. 
"""
function checkfrobnorm!(
    V :: AbstractVector{Matrix{T}}
) where {T <: Real}

    frobnorm = 0
    for Vi in V
        frobnorm = norm(Vi) 
        if frobnorm != 1
            Vi ./= frobnorm
        end
    end 
end 
"""
    rankvarcomps(σ2path; tol=1e-6)

# Input
- `σ2path`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`

# Keyword 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6 

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    σ2path :: AbstractMatrix{T};
    tol    :: Float64=1e-6
    ) where {T <: Real}

    # size of solution path 
    novarcomp, nlambda = size(σ2path)

    # initialize array for ranking 
    ranking = Int[]

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        tmp = findall(x -> x > tol, view(σ2path, 1:(novarcomp-1), col))
        for j in tmp 
            if !(j in ranking)
                push!(ranking, j)
            end
        end
    end 
    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:(novarcomp-1), ranking)
    rest = [rest; novarcomp]

    return ranking, rest 
end 

"""
    rankvarcomps(Σpath; tol=1e-6)

# Input
- `Σpath`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect`, `vcselectpath`

# Keyword 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-6 

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    Σpath :: AbstractMatrix{T};
    tol    :: Float64=1e-6
    ) where {T <: Real}

    # size of solution path 
    novarcomp, nlambda = size(σ2path)

    # initialize array for ranking 
    ranking = Int[]

    # go through solution path and find the order in which variance component enters
    for col in nlambda:-1:2
        tmp = findall(x -> x > tol, view(σ2path, 1:(novarcomp-1), col))
        for j in tmp 
            if !(j in ranking)
                push!(ranking, j)
            end
        end
    end 
    # rest of the variance components that are estimated to be zero at all λ > 0
    rest = setdiff(1:(novarcomp-1), ranking)
    rest = [rest; novarcomp]

    return ranking, rest 
end 

"""
    plotsolpath(σ2path, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of solution path at varying λ values. Use backend such as `gr()`.

# Input
- `σ2path`: solution path (in numeric matrix) to be plotted, each column should 
        represent variance components at specific λ 
        as in output from `vcselect`, `vcselectpath`
- `λpath`: vector of tuning parameter λ values 

# Keyword 
- `title`: title of the figure, default is "Solution Path"
- `xlab`: x-axis label, default is minimum of λpath
- `ylab`: y-axis label, default is maximum of λpath
- `linewidth`: line width, default is 1.0

# Output 
- plot of solution path 
"""
function plotsolpath(
    σ2path    :: AbstractMatrix{T},
    λpath     :: AbstractVector{T};
    title     :: AbstractString = "Solution Path",
    xlab      :: AbstractString = "\\lambda",
    xmin      :: AbstractFloat = minimum(λpath),
    xmax      :: AbstractFloat = maximum(λpath),
    ylab      :: AbstractString = "\\sigma^2",
    linewidth :: AbstractFloat = 1.0
) where {T <: Real}

    # size of solution path 
    novarcomp, nlambda = size(σ2path)

    # get ranking of variance components
    ranking, rest = rankvarcomps(σ2path)

    # transpose solpath s.t. each row is estimates at particular lambda
    tr_σ2path = σ2path'

    # label in the permuted order 
    legendlabel = "\\sigma^{2}[$(ranking[1])]"
    for i in ranking[2:end]
        legendlabel = hcat(legendlabel, "\\sigma^{2}[$i]")
    end
    for i in 1:length(rest)
        legendlabel = hcat(legendlabel, "")
    end 

    # plot permuted solution path (decreasing order)
    plot(λpath, tr_σ2path[:, [ranking; rest]], label=legendlabel, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth, legendtitle="ranking")
    title!(title)     

end 

"""
    matarray2mat(matarray)

Convert array of matrices to single matrix. All matrices in `matarray` should have 
the same dimension. e.g. Σpath

# Input 
- `matarray`: array of matrices to be converted 

# Output 
- `mat`: single matrix containing all matrices in `Σpath`
"""
function matarray2mat(matarray)
  
    nvarcomps, nlambda = size(matarray)
    d = size(matarray[1, 1], 1)
    mat = zeros(nvarcomps * d, nlambda * d)
  
    for r in 1:nvarcomps
      for c in 1:nlambda 
        mat[(r * d - d + 1):(r * d), (c * d - d + 1):(c * d)] = matarray[r, c]
      end 
    end 
  
    return mat 
end 