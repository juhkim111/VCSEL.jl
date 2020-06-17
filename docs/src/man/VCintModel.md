# VCintModel

We have the following variance components model: 

```math
Y = X\beta + Z_1\gamma_1 + E Z_1 \alpha_1 +  \ldots + Z_m\gamma_m + E Z_m \alpha_m + \epsilon  \hspace{8em} (1)
```
where 

* ``Y``: ``n\times 1`` vector of continuous response  
* ``X``: ``n\times p`` known design matrix for fixed effects 
* ``Z_i``: ``n\times q_i`` known design matrix with corresponding random effects ``\gamma_i``
* ``\beta``: ``p \times 1`` unknown vector of fixed effects parameters 
* ``\gamma_i``: ``q_i \times 1`` unknown vector of random main effects parameters with ``\gamma_i \sim \text{Normal}(0, \sigma^2_{i1} I_{q_i})``
    * ``\sigma^2_{i1}``: $i$-th main effect variance component, ``i=1,\ldots, m``
* ``\alpha_i``: ``q_i \times 1`` unknown vector of random interaction effects parameters with ``\alpha_i \sim \text{Normal}(0, \sigma^2_{i2} I_{q_i})``
    - ``\sigma^2_{i2}``: $i$-th interaction effect variance component, ``i=1,\ldots, m``
* ``E``: ``n\times n`` diagonal matrix whose diagonal entries are factors that interact with each main effect (``Z_i``). 
* ``\epsilon``: ``n\times 1`` vector of errors with ``\epsilon \sim \text{Normal}(0, \sigma^2_0 I_n)``
    - ``\sigma^2_0``: residual or intercept variance component

Equivalently, we can write (1) as


```math
Y \sim \text{Normal}(X\beta, \sigma^2_{11} V_{11} +  \sigma^2_{12} V_{12}+ \cdots + \sigma^2_{m1} V_{m1} + \sigma^2_{m2} V_{m2} + \sigma^2_0 V_0),  \hspace{3em} (2)
```

where ``V_{i1} = Z_{i1} Z_{i1}^T`` and ``V_{i2} = E Z_{i1} Z_{i1}^T E^T.``



In the equation (2), 

* ``Y``, ``X``, and ``V_{11},V_{12},...,V_{m1},V_{m2}, V_0`` form the **data** and 

* ``\beta`` and ``\sigma^2_{11},\sigma^2_{12}...,\sigma^2_{m1},\sigma^2_{m2},\sigma_0^2`` are **parameters**. 



## Goal

We want to identify sets of variance components (``\sigma^2_{11},\sigma^2_{12}``) that are associated with response $Y$. This can be achieved by VCSEL.

By selecting or not selecting pairs of variance components (main effect and interaction effect variance components), we obtain main variance components associated with response while accounting for interaction effect. 

## Steps

### Step 0: Load the package

If not installed, follow the [Installation](@ref) instruction on main page. Then load the package:

```julia
using VCSEL
```

### Step 1: Construct a model with data 

Construct an instance of [`VCintModel`](@ref), which requires users to supply 

* `Y`: `n x 1` response vector 
* `X`: `n x p` covariate matrix (if exists)
* `V=[V[1],...,V[m],V[m+1]]`: a vector of `m+1` `n x n` covariance matrices  
* `Vint=[Vint[1],...,Vint[m]]`: a vector of `m` `n x n` covariance matrices.

Example: 

```julia 
# initialize VCintModel instance 
vcm1 = VCintModel(Y, X, V, Vint)
vcm2 = VCintModel(Y, V, Vint) # if there's no covariate matrix 
```

`VCintModel` also has the following fields for its parameters: 

* `B`: `p x d` mean regression coefficients 
* `Σ = [Σ[1],...,Σ[m],Σ[m+1]]`: variance component parameters for main effects (`Σ[1],...,Σ[m]`) and intercept (`Σ[m+1]`)
* `Σint = [Σint[1],...,Σint[m]]`: variance component parameters for interaction effects.

By default, the vector of variance component parameters are initialized to be vectors of ones (e.g. `ones(length(V))`, `ones(length(Vint))`). Users can set initial values of their choice in this step if they wish to. 

Example:

```julia
Σ = fill(0.5, length(V))
Σint = fill(0.5, length(Vint))
vcm3 = VCModel(Y, X, V, Vint, Σ, Σint)
vcm4 = VCModel(Y, V, Vint, Σ, Σint)
```

### Step 2: Fit model 

Call optimization routine [`vcselect!`](@ref) to select variance components at a given tuning parameter $\lambda$ with some penalty (options: [`NoPenalty()`, `L1Penalty()`, `MCPPenalty()`](https://github.com/JuliaML/PenaltyFunctions.jl#Element-Penalties)).


Examples:

```julia
# fit model with lasso (L1) penalty at tuning parameter λ=1.5
vcselect!(vcm1; penfun=L1Penalty(), λ=1.5)
# fit model with MCP penalty at tuning parameter λ=5.5
vcselect!(vcm2; penfun=MCPPenalty(), λ=5.5)
```

If penalty function is given but tuning parameter  𝜆  is not given,  𝜆  is set to 1.0.

Example: 

```julia 
# following commands are equivalent 
vcselect!(vcm3; penfun=L1Penalty()) 
vcselect!(vcm3; penfun=L1Penalty(), λ=1.0) 
```

If no penalty function is given, it fits model without any penalty, which is same as penfun=NoPenalty() or λ=0.

Example:

```julia
# following commands are equivalent 
vcselect!(vcm4)
vcselect!(vcm4; penfun=NoPenalty())
vcselect!(vcm4; λ=0)
```

Estimated parameters can be accessed using the `vcm.Σ` notation.

Example:

```julia
# variance components for main effects
vcm1.Σ
# variance components for interaction effects
vcm1.Σint
# mean effects
vcm1.β
```

### Step 2 Alternative: Get solution path 

If you want to fit a model over a grid of tuning parameter $\lambda$ values (i.e. obtain solution path), use `vcselectpath!`.

For details about the function, go to [`vcselectpath!`](@ref) in API page. 

If we only supply `VCintModel` instance when calling `vcselectpath!`, it returns the same output as `vcselect!` with `penfun=NoPenalty()`.

Here we call `vcselectpath!` with penfun=`L1Penalty()`. Since we do not provide `nλ` or `λpath`, a grid of 100 $\lambda$ values is generated internally.

Example:

```julia
vcm = VCModel(Y, X, V, Vint)
Σ̂path, Σ̂intpath, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; 
    penfun=L1Penalty());
```
