"""
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
- `Σpath`: solution path (matrix of matrices), each column should 
    contain estimated variance components at specific λ 
    as in output from `vcselectpath!`
- `resvarcomp`: logical flag indicating the presence of residual variance component 
    in `Σpath`. If `true`, last row of `Σpath` is ignored. Default is `true`

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
    rankvarcomps(Σpath, Σpath2; tol=1e-6, resvarcomp=true, resvarcomp2=true)

Obtain rank of variance components from a paired solution paths, e.g. `Σpath` and `Σpath2`.   
Ranks are calculated using norm of paired variance components. 
# Input
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
# Output 
- `ranks`: rank of each variance component based on the order in which it enters 
    solution path
- `rest`: rest of the variance components that are estimated to be zero at all λ > 0
"""
function rankvarcomps(
    Σpath      :: AbstractMatrix{T},
    Σpath2     :: AbstractMatrix{T};
    tol         :: Float64 = 1e-6,
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
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
    ) where {T <: Real}

    # size of solution path 
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
- plot of solution path 
"""
function plotsolpath(
    Σpath       :: AbstractMatrix{T},
    Σpath2      :: AbstractMatrix{T},
    λpath       :: AbstractVector{T};
    title       :: AbstractString = "Solution Path",
    xlab        :: AbstractString = "\$\\lambda\$",
    xmin        :: AbstractFloat = minimum(λpath),
    xmax        :: AbstractFloat = maximum(λpath),
    ylab        :: AbstractString = "\$||(\\sigma_{1}^2, \\sigma_{2}^2)||_2\$",
    nranks      :: Int = size(Σpath, 1),
    linewidth   :: AbstractFloat = 1.0, 
    legend      :: Bool = true,
    legendout   :: Bool = false,
    legendtitle :: AbstractString = "Ranking",
    resvarcomp  :: Bool = true,
    resvarcomp2 :: Bool = false
) where {T <: Real}

    # error handling 
    @assert size(Σpath, 2) == size(Σpath2, 2) "both solution paths must have the same number of lambdas!\n"

    # size of solution path 
    m, nlambda = size(Σpath2)

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

        end 

        for i in 1:(m + 1 - nranks)
            legendlabel = hcat(legendlabel, "")
        end 

        # plot permuted solution path (decreasing order)
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
            l = @layout [b c{0.13w}]
            plot(pt1, pt2, layout=l)
        end 
       
    # no legend 
    else 
        plot(λpath, trnormpath[:, [ranking; rest]], legend=false, 
        xaxis=(xlab, (xmin, xmax)), yaxis=(ylab), width=linewidth)
        title!(title)     
    end 

end 


