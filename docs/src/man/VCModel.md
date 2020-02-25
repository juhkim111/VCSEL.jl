# VCModel  

We have the following variance component model
 
```math
\text{vec}(Y) \sim \text{Normal}(XB, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m + \Sigma_0 \otimes V_0),  \hspace{8em} (1)
```

where 

* $Y$: $n\times d$ response matrix 
* $X$: $n\times p$ covariate matrix 
* $V = (V_1,...,V_m,V_0)$: a vector of $m+1$ $n \times n$ covariance matrices

form the data and 

* $B$: $p \times d$ mean parameter matrix that explains the effect of covariates $X$ on response $Y$
* $\Sigma = (\Sigma_1,...,\Sigma_m,\Sigma_0)$: a vector of $m+1$ $d \times d$ variance components

are parameters. 


If $d=1$, (1) boils down to 

```math
y \sim \text{Normal}(X\beta, \sigma_1^2 V_1 + \cdots + \sigma_m^2 V_m + \sigma_0^2 I_n ).
```

Suppose we want to identify variance components that are associated with $Y.$ This can be achieved by **VCSEL** algorithm, an MM algorithm that selects relevant variance components via penalization method. To find the estimates of parameters $(B, \Sigma_1, \ldots, \Sigma_m, \Sigma_0),$ we take 2 steps: 



**Step 1 (Construct a model with data)**. Construct an instance of `VCModel`, which consists of fields 

* `Y`: `n x d` response matrix 
* `X`: `n x p` covariate matrix 
* `V = (V1,...,Vm,V0)`: a vector of `m+1` `n x n` covariance matrices. 


**Step 2 (Fit model)**.  
