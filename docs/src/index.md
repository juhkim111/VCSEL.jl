# VarianceComponentSelect.jl

**VarianceComponentSelect** is a Julia package for implementing MM algorithm that selects relevant variance components via penalization method. Variance component model takes the form 

\begin{eqnarray}
\text{vec}(Y) \sim \text{Normal}(XB, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m + \Sigma_0 \otimes I_n )  \hspace{8em} (1)
\end{eqnarray}

where $V_i$ has [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) 1 for all $i$ and $\otimes$ is the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

In this model, **data** is represented by 

* `Y`: `n x d` response matrix 
* `X`: `n x p` covariate matrix 
* `V = (V1,...,Vm,V0)`: a vector of `m+1` `n x n` covariance matrices, $V_0$

and **parameters** are 

* `B`: `p x d` mean parameter matrix
* `Σ=(Σ1,...,Σm,Σ0)`: a vector of `m+1` `d x d` variance components.

If `d=1`, (1) boils down to 
$$y \sim \text{Normal}(X\beta, \sigma_1^2 V_1 + \cdots + \sigma_m^2 V_m + \sigma_0^2 I_n ).$$

## Installation 

This package requires Julia v0.7.0 or later, which can be obtained from
https://julialang.org/downloads/ or by building Julia from the sources in the
https://github.com/JuliaLang/julia repository.

The package has not yet been registered and must be installed using the repository location.
Start julia and use the `]` key to switch to the package manager REPL

```julia
(v1.1) pkg> add https://github.com/juhkim111/VarianceComponentSelect.jl
```

Use the backspace key to return to the Julia REPL.
