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
* ``\gamma_i``: ``q_i \times 1`` unknown vector of random main effects parameters with ``\gamma_i \sim \text{Normal}(0, \Sigma_{i1} I_{q_i})``
* ``\alpha_i``: ``q_i \times 1`` unknown vector of random interaction effects parameters with ``\alpha_i \sim \text{Normal}(0, \Sigma_{i2} I_{q_i})``
* ``E``: ``n\times n`` diagonal matrix whose diagonal entries are factors that interact with each main effect (``Z_i``). 
* ``epsilon``: ``n\times 1`` vector of errors with ``\epsilon \sim \text{Normal}(0, \Sigma_0 I_n)``

Equivalently, we can write (1) as


```math
Y \sim \text{Normal}(X\beta, \Sigma_{11} V_{11} +  \Sigma_{12} V_{12}+ \cdots + \Sigma_{m1} V_{m1} + \Sigma_{m2} \otimes V_{m2} + \Sigma_0 \otimes V_0),  
```

where ``V_{i1} = Z_{i1} Z_{i1}^T`` and ``V_{i2} = E Z_{i1} Z_{i1}^T E^T.``



In the equation (1), 

* ``Y``: ``n\times 1`` response vector 
* ``X``: ``n\times p`` covariate matrix 
* ``V = (V_{11},V_{12},...,V_{m1},V_{m2}, V_0)``: a vector of $2m+1$ $n \times n$ covariance matrices

form the data and 

* ``\beta``: ``p \times 1`` mean parameter vector that explains the effect of covariates $X$ on response $Y$
* ``\Sigma = (\Sigma_{11},\Sigma_{12}...,\Sigma_{m1},\Sigma_{m2},\Sigma_0)``: a vector of $2m+1$ variance components 
    - ``\Sigma_{i1}``: main effect variance component for $i$-th group, ``i=1,\ldots, m``
    - ``\Sigma_{i2}``: interaction effect variance component for $i$-th group, ``i=1,\ldots, m``
    - ``\Sigma_0``: residual variance component

are parameters. 



## Goal

We want to identify sets of variance components (``\Sigma_{11},\Sigma_{12}``) that are associated with response $Y$. This can be achieved by VCSEL.

By selecting or not selecting pairs of variance components (main effect and interaction effect variance components), we obtain main variance components associated with response while accounting for interaction effect. 

## Steps

### Step 0: Load the package

```julia
using VCSEL
```

### Step 1: Construct a model with data 

Construct an instance of `VCModel`, which requires users to supply 

* `Y`: `n x 1` response vector 
* `X`: `n x p` covariate matrix (if exists)
* `V=[V[1],...,V[m],V[m+1]]`: a vector of `m+1` `n x n` covariance matrices  
* `Vint=[Vint[1],...,Vint[m]]`: a vector of `m` `n x n` covariance matrices.

```julia 
# initialize VCintModel instance 
vcm1 = VCintModel(Y, X, V, Vint)
vcm2 = VCintModel(Y, V, Vint) # if there's no covariate matrix 
```

`VCintModel` also has the following fields for its parameters: 

* `B`: `p x d` mean regression coefficients 
* `Σ = [Σ[1],...,Σ[m],Σ[m+1]]`: variance component parameters for main effects (`Σ[1],...,Σ[m]`) and residual (`Σ[m+1]`)
* `Σint = [Σint[1],...,Σint[m]]`: variance component parameters for interaction effects.

By default, the vector of varaince component parameters are initialized to be a vector of identity matrices (e.g. `[Matrix(1.0*I, d, d) for i in 1:(m+1)]`). Users can set initial values of variance component parameters in this step if they wish to. 

Example:

```julia
vcm3 = VCModel(Y, X, V, Vint, Σ, Σint)
vcm4 = VCModel(Y, V, Vint, Σ, Σint)
```

### Step 2: Fit model 

Call optimization routine `vcselect!` to select variance components at a given tuning parameter $\lambda$ with some penalty (options: `NoPenalty()`, `L1Penalty()`, `MCPPenalty()`).


Required input argument for executing `vcselect!` is `VCModel`:
    
- `VCintModel`.

Keyword arguments for `vcselect!` are:


- `penfun`: penalty function. Default is `NoPenalty()`. Other options are `L1Penalty()` and `MCPPenalty()`.
- `λ`: tuning parameter. Default is 1.0.    
- `penwt`: penalty weights. Default is (1,...1,0).
- `standardize`: logical flag for covariance matrix standardization. Default is `false`.
    If true, `V[i]` and `Vint[i]` are standardized by its Frobenius norm, and parameter estimates are 
    returned on the original scale.
- `maxiters`: maximum number of iterations. Default is 1000.
- `tol`: convergence tolerance. Default is `1e-5`.
- `verbose`: display switch. Default is false.
- `checktype`: check argument type switch. Default is true.

Examples:

```julia
vcselect!(vcm1; penfun=L1Penalty(), λ=1.5)
vcselect!(vcm2; penfun=MCPPenalty(), λ=5.5)
```

If penalty function is given but tuning parameter  𝜆  is not given,  𝜆  is set to 1.0.

```julia 
# following commands are equivalent 
vcselect!(vcm3; penfun=L1Penalty()) 
vcselect!(vcm3; penfun=L1Penalty(), λ=1.0) 
```

If no penalty function is given, it fits model without any penalty, which is same as penfun=NoPenalty() or λ=0.

```julia
# following commands are equivalent 
vcselect!(vcm4)
vcselect!(vcm4; penfun=NoPenalty())
vcselect!(vcm4; λ=0)
```


```julia

```
