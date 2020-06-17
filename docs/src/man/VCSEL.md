# What is VCSEL? 

VCSEL is an [Majorization-Minimization (MM) algorithm](https://en.wikipedia.org/wiki/MM_algorithm) that selects variance components that are associated with response variable(s) via penalization method. 

## What are variance components?

Suppose we have the following mixed model: 

```math
Y = X\beta + Z\gamma + \epsilon 
```

where 

* ``Y``: ``n\times 1`` vector of responses 
* ``X``: ``n \times p`` known design matrix for the fixed effects 
* ``Z``: ``n \times q`` known design matrix for the random effects 
* ``\beta``: ``p \times 1`` vector of unknown fixed effects  
* ``\gamma``: ``q \times 1`` vector of unknown random effects with ``\gamma \sim \text{Normal}(0, \sigma_{\gamma}^2 I_q)``
* ``\epsilon``: $n\times 1$ vector of unknown random errors with $\epsilon \sim \text{Normal}(0, \sigma_{\epsilon}^2 I_n)$ 
* ``\gamma`` and ``\epsilon`` are independent.

Equivalently, we can write 

```math 
Y \sim \text{Normal}(X\beta, \sigma_{\gamma}^2 Z Z^T + \sigma_{\epsilon}^2)
```
which states variance of the dependent variable $Y$ (aka $\text{Var}(Y)$) equals to $\sigma_{\gamma}^2 Z Z^T + \sigma_{\epsilon}^2$. 

Because $\sigma_{\gamma}^2$ and $\sigma_{\epsilon}^2$ contribute to the variance of the dependent variable $Y$, they are called __variance components__. 

## Testing for zero variance component

Suppose one wants to test whether the contribution due to the random components is significant. Then we can set up the null hypothesis 

```math
H_0: \gamma = 0
```
which indicates that the random components of the model (except the error term) are not significant, hence a fixed-effects model is sufficient to explain the linear relationship between the outcome variables and covariates.

Note that testing ``H_0: \gamma = 0`` is equivalent to testing if the variance component is zero

```math
H_0: \sigma_{\gamma}^2 = 0
```
which requires fewer parameters to estimate. 





## What if there are multiple variance components? 

Suppose that you have multiple random effects vectors and that you want to find which of those are associated with the response variable. Then you can jointly model all random effects:

```math
Y = X\beta + Z_1\gamma_1 + \cdot + Z_m \gamma_m + \epsilon 
```

where 

* ``Z_i``: ``n \times q_i`` known design matrix for the random effects where $i=1,\dots, m$
* ``\gamma_i \sim \text{Normal}(0, \sigma_i^2 I_{q_i}), i=1,\ldots, m``
* ``\epsilon \sim \text{Normal}(0, \sigma_{\epsilon}^2 I_n)``.

## VCSEL

Now this is were VCSEL algorithm comes in. VCSEL implements [Majorization-Minimization (MM) algorithm](https://en.wikipedia.org/wiki/MM_algorithm) to select variance components that are relevant to the response $Y$. This algorithm minimizes the negative log-likelihood of the model plus a penalty using a surrogate function. For details, please see our paper. 

## `VCSEL.jl` features

`VCSEL.jl` package can handle 

* univariate response model 

  ```math 
  Y \sim \text{Normal}(X\beta, \sigma_1^2 V_1 + \cdot + \sigma_m^2 V_m + \sigma_{\epsilon}^2 I_n
  ```

  + ``Y``: $n\times 1$ response vector 
  + ``V_i ,i=1,\ldots, m``: covariance matrices corresponding to each random effects vector (e.g. ``V_i = Z_i Z_i^T``). 

* multivariate response model 

  ```math 
   Y \sim \text{Normal}(X\beta, \Sigma_1 \otimes V_1 + \cdot + \Sigma_m \otimes V_m + \Sigma_{\epsilon} \otimes I_n
  ```

  + ``Y``: ``n\times d`` response matrix
  + ``\Sigma_i, i=1,\ldots, m``: $d\times d$ variance component matrices
  + ``\otimes``: [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

* univariate response model with interaction terms 

  ```math 
  Y \sim \text{Normal}(X \beta, \sigma_{11}^2 V_{11} + \sigma_{12}^2 V_{12} + \cdot + \sigma_{m1}^2 V_{m1} + \sigma_{m2}^2 V_{m2} + \sigma_{\epsilon}^2 I_n)
  ```

   + ``Y``: ``n \times 1`` response vector 
   + ``\sigma_{i1}^2`` and ``\sigma_{i2}^2``: pair of variance components that are selected/unselected together (``i=1,\ldots, m``)	
      - ``\sigma_{i1}^2`` represents variance component for main effects while ``\sigma_{i2}^2`` represents variance component for interaction effects.

