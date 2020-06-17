# VCModel

We have the following variance components model: 

```math
\text{vec}(Y) \sim \text{Normal}(XB, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m + \Sigma_0 \otimes V_0),  \hspace{8em} (1)
```

where $\text{vec}(\cdot)$ stacks up columns of the given matrix on top of one another.


In the equation (1), 

* ``Y``: ``n\times d`` response matrix 
* ``X``: ``n\times p`` covariate matrix 
* ``V = (V_1,...,V_m,V_0)``: a vector of $m+1$ $n \times n$ covariance matrices

form the data and 

* ``B``: ``p \times d`` mean parameter matrix that explains the effect of covariates $X$ on response $Y$
* ``\Sigma = (\Sigma_1,...,\Sigma_m,\Sigma_0)``: a vector of $m+1$ $d \times d$ variance components matrices 

are parameters. 




If $Y$ is a $n \times 1 $ vector (i.e. $d=1$), (1) boils down to 

```math
Y \sim \text{Normal}(XB, \Sigma_1 V_1 + \cdots + \Sigma_m V_m + \Sigma_0 V_0), 
```

where 

* ``B``: ``p \times 1`` mean parameter vector 
* ``\Sigma_i, i=1,\ldots, m`` represent non-negative scalar variance components. 




## Goal 

Suppose we want to identify variance components that are associated with $Y.$ This can be achieved by **VCSEL** algorithm, an MM algorithm that selects relevant variance components via penalization method. 

To find the estimates of parameters $(B, \Sigma_1, \ldots, \Sigma_m, \Sigma_0),$ we take 2 steps.




## Steps 

### Step 0: Load the package

Assuming the package is already installed, load the package:

```julia 
using VCSEL
```

If not installed, follow the [Installation](@ref) instruction on main page. 

### Step 1: Construct a model with data

Construct an instance of [`VCModel`](@ref), which requires users to supply

* `Y`: `n x d` response matrix 
* `X`: `n x p` covariate matrix (if exists)
* `V = [V1,...,Vm,V0]`: a vector of `m+1` `n x n` covariance matrices. 

Example: 

```julia 
# initialize VCModel instance
vcm1 = VCModel(Y, X, V)
vcm2 = VCModel(Y, V) # if there's no covariate matrix 
```

`VCModel` also has the following fields for its parameters: 

* `B`: `p x d` mean regression coefficients 
* `Σ = [Σ[1],...,Σ[m],Σ[0]]$`: variance component parameters.

By default, the vector of varaince component parameters are initialized to be a vector of identity matrices (e.g. `[Matrix(1.0*I, d, d) for i in 1:(m+1)]`). Users can set initial values of variance component parameters in this step if they wish to. 

Example:

```julia 
# matrix of ones as initial estimates 
Σ = [ones(d,d) for i in 1:(m+1)])
# initialize VCModel instance with initial estimates of variance components
vcm3 = VCModel(Y, X, V, Σ)
vcm4 = VCModel(Y, V, Σ)
```

### Step 2: Fit model

Call optimization routine [`vcselect!`](@ref) to select variance components at a given tuning parameter $\lambda$ with some penalty (options: [`NoPenalty()`, `L1Penalty()`, `MCPPenalty()`](https://github.com/JuliaML/PenaltyFunctions.jl#Element-Penalties)).
For details about the function, go to [`vcselect!`](@ref) in API page.

Example: 

```julia 
# fit model with lasso (L1) penalty at tuning parameter λ=1.5
vcselect!(vcm1; penfun=L1Penalty(), λ=1.5)
# fit model with MCP penalty at tuning parameter λ=5.5
vcselect!(vcm2; penfun=MCPPenalty(), λ=5.5)
```

If penalty function is given but tuning parameter $\lambda$ is not given, $\lambda$ is set to 1.0. 

Example:

```julia 
# following commands are equivalent 
vcselect!(vcm3; penfun=L1Penalty()) 
vcselect!(vcm3; penfun=L1Penalty(), λ=1.0) 
```

If no penalty function is given, it fits model without any penalty, which is same as `penfun=NoPenalty()` or `λ=0`.

Example: 

```julia 
# following commands are equivalent 
vcselect!(vcm4)
vcselect!(vcm4; penfun=NoPenalty())
vcselect!(vcm4; λ=0)
```

### Step 2 Alternative: Get solution path

If you want to fit a model over a grid of tuning parameter $\lambda$ values (i.e. obtain solution path), use `vcselectpath!`.

For details about the function, go to [`vcselectpath!`](@ref) in API page.



If we only supply `VCModel` instance when calling `vcselectpath!`, it returns the same output as `vcselect!` with `penfun=NoPenalty()`. 



Here we call `vcselectpath!` with `penfun=L1Penalty()`. Since we do not provide `nλ` or `λpath`, a grid of 100 $λ$ values is generated internally. 

Example:

```julia 
vcm = VCModel(Y, X, V)
Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; 
    penfun=L1Penalty());
```

Or we can create a $\lambda$ grid of our own.

Example:

```julia
vcm = VCModel(Y, X, V)
λpath = range(0, 10, length=50)
Σ̂path, β̂path,  = vcselectpath!(vcm; penfun=L1Penalty(), λpath=λpath)
```

### Summarise/visualize results
