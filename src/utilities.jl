export plotsolpath, rankvarcomps, matarray2mat

"""
    rankvarcomps(σ2path; tol=1e-8)

# Input
- `σ2path`: solution path (in numeric matrix), each column should 
    represent estimated variance components at specific λ 
    as in output from `vcselect!`, `vcselectpath!`
- `resvarcomp`: logical flag indicating the presence of residual variance component 
    in `σ2path`. If `true`, last row of `σ2path` is ignored. Default is `true`

# Keyword Argument 
- `tol`: a variance component less than `tol` is considered zero, default is 1e-8 

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    σ2path     :: AbstractMatrix{T};
    tol        :: Float64 = 1e-8,
    resvarcomp :: Bool = true
    ) where {T <: Real}

    # size of solution path 
    novarcomps, nlambda = size(σ2path)

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
        idx = findall(x -> x > tol, view(σ2path, 1:m, col))
        sortedidx = sortperm(σ2path[idx], rev=true)
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
- `Σpath`: solution path (matrix of matrices), each column should 
    contain estimated variance components at specific λ 
    as in output from `vcselectpath!`
- `resvarcomp`: logical flag indicating the presence of residual variance component 
    in `σ2path`. If `true`, last row of `σ2path` is ignored. Default is `true`

# Keyword Argument 
- `tol`: a variance component whose `p`-norm is less than `tol` is considered zero, 
    default is 1e-8 
- `p`: `p` for determining which norm to use, default is 2. See [`norm(A, p::Real=2)`](@ref)

# Output 
- `ranking`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    Σpath      :: AbstractMatrix{Matrix{T}};
    tol        :: Float64 = 1e-8,
    p          :: Real = 2,
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

"""
    plotsolpath(σ2path, λpath; title="Solution Path", xlab="λ", ylab="σ2", 
            xmin=minimum(λpath), xmax=minimum(λpath), tol=1e-6)

Output plot of solution path at varying λ values. Use backend such as `gr()`.

# Input
- `σ2path`: solution path (in numeric matrix) to be plotted, each column should 
        represent variance components at specific λ as in output from `vcselectpath!`. 
	The last row of `σ2path` should be for residual variance component. Otherwise set `resvarcomp=false` to indicate the absence. 
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
- `resvarcomp`: logical flag for indicating residual variance component in `σ2path`. 	   
      Default is true. 

# Output 
- plot of solution path 
"""
function plotsolpath(
    σ2path     :: AbstractMatrix{T},
    λpath      :: AbstractVector{T};
    title      :: AbstractString = "Solution Path",
    xlab       :: AbstractString = "\\lambda",
    ylab       :: AbstractString = "\\sigma^2",
    xmin       :: AbstractFloat = minimum(λpath),
    xmax       :: AbstractFloat = maximum(λpath),
    linewidth  :: AbstractFloat = 1.0,
    nranks     :: Int = size(σ2path, 1),
    legend     :: Bool = true,
    legendout  :: Bool = true, 
    resvarcomp :: Bool = true
    ) where {T <: Real}

    # size of solution path 
    novarcomps, nlambda = size(σ2path)

    # get ranking of variance components
    ranking, rest = rankvarcomps(σ2path; resvarcomp=resvarcomp)

    # transpose solpath s.t. each row is estimates at particular lambda
    tr_σ2path = σ2path'

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

        for i in 1:(nvarcomps - nranks)
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)

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
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
        plot(λpath, tr_σ2path[:, [ranking; rest]], legend=false, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 
   

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


