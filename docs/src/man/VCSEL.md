<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#What-are-variance-components?" data-toc-modified-id="What-are-variance-components?-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>What are variance components?</a></span></li><li><span><a href="#Testing-for-zero-variance-component" data-toc-modified-id="Testing-for-zero-variance-component-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Testing for zero variance component</a></span></li><li><span><a href="#What-if-there-are-multiple-variance-components?" data-toc-modified-id="What-if-there-are-multiple-variance-components?-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>What if there are multiple variance components?</a></span></li><li><span><a href="#VCSEL" data-toc-modified-id="VCSEL-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>VCSEL</a></span></li><li><span><a href="#VCSEL.jl-features" data-toc-modified-id="VCSEL.jl-features-5"><span class="toc-item-num">5&nbsp;&nbsp;</span><code>VCSEL.jl</code> features</a></span></li></ul></div>

# What is VCSEL? 

VCSEL is an [Majorization-Minimization (MM) algorithm](https://en.wikipedia.org/wiki/MM_algorithm) that selects variance components that are associated with response variable(s) via penalization method. 

## What are variance components?

Suppose we have the following mixed model: 

```math
Y = X\beta + Z\gamma + \epsilon 
```

where 

* $Y$ is an $n\times 1$ vector of responses 
* $X$ is the $n \times p$ known design matrix for the fixed effects 
* $Z$ is the $n \times q$ known design matrix for the random effects 
* $\beta$ is a $p \times 1$ vector of unknown fixed effects  
* $\gamma$ is a $q \times 1$ vector of unknown random effects with $\gamma \sim \text{Normal}(0, \sigma_{\gamma}^2 I_q)$
* $\epsilon$ is an $n\times 1$ vector of unknown random errors with $\epsilon \sim \text{Normal}(0, \sigma_{\epsilon}^2 I_n)$ 
* $\gamma$ and $\epsilon$ are independent.

Equivalently, we can write 

```math 
Y \sim \text{Normal}(X\beta, \sigma_{\gamma}^2 Z Z^T + \sigma_{\epsilon}^2)
```
which means variance of the dependent variable $Y$ (aka $\text{Var}(Y)$) equals to $\sigma_{\gamma}^2 Z Z^T + \sigma_{\epsilon}^2$. 

Because $\sigma_{\gamma}^2$ and $\sigma_{\epsilon}^2$ contribute to the variance of the dependent variable $Y$, they are called variance components. 

## Testing for zero variance component

Suppose one wants to test whether the random effects is significant. Then we can set up the null hypothesis $H_0: \gamma = 0$, which indicates that $Z$ has no effect on the mean of $Y$. 

If $q$ (the number of elements in $\gamma$) is large, however, we have too many parameters to estimate, running into limited power issue. 

Instead, we can test if the variance component is zero

$$H_0: \sigma_{\gamma}^2 = 0,$$

which is equivalent to testing $H_0: \gamma = 0$.




## What if there are multiple variance components? 

Suppose that you have multiple random effects vector and that you want to find which random effects are associated with the response variable. Then you can jointly model all random effects:

```math
Y = X\beta + Z_1\gamma_1 + \cdot + Z_m \gamma_m + \epsilon 
```

where 

* $Z_i$ is the $n \times q_i$ known design matrix for the random effects where $i=1,\dots, m$
* $\gamma_i \sim \text{Normal}(0, \sigma_i I_{q_i}), i=1,\ldots, m$
* $\epsilon \sim \text{Normal}(0, \sigma_{\epsilon} I_n)$.

## VCSEL

Now this is were VCSEL algorithm comes in. VCSEL implements [Majorization-Minimization (MM) algorithm](https://en.wikipedia.org/wiki/MM_algorithm) to select variance components that are relevant to the response $Y$.

Under the hood, VCSEL algorithm minimizes the negative log-likelihood of the model plus a penalty using a surrogate function. For details, please see our paper. 

## `VCSEL.jl` features

`VCSEL.jl` package can handle 

* univariate response model 

```math 
Y \sim \text{Normal}(X\beta, \sigma_1^2 V_1 + \cdot + \sigma_m^2 V_m + \sigma_{\epsilon}^2 I_n
```
where $Y$ is an $n\times 1$ vector and $V_i ,i=1,\ldots, m$ are covariance matrices corresponding to each random effects vector (e.g. $Z Z^T$ in the notation above). 

* multivariate response model 

```math 
Y \sim \text{Normal}(X\beta, \Sigma_1 \otimes V_1 + \cdot + \Sigma_m \otimes V_m + \Sigma_{\epsilon} \otimes I_n
```

where $Y$ is an $n\times d$ matrix, $\Sigma_i, i=1,\ldots, m$ are $d\times d$ variance component matrices, and $\otimes$ is a [Kronecker matrix](https://en.wikipedia.org/wiki/Kronecker_product).  

* univariate response model with interaction terms 

```math 
Y \sim \text{Normal}(X \beta, \sigma_{11}^2 V_{11} + \sigma_{12}^2 V_{12} + \cdot +  \sigma_{m1}^2 V_{m1} + \sigma_{m2}^2 V_{m2} + \sigma_{\epsilon}^2 I_n)
```

where $Y$ is an $n \times 1$ vector and $\sigma_{i1}^2$ and $\sigma_{i2}^2$ are a pair of variance components that are selected/unselected together ($i=1,\ldots, m$). $\sigma_{i1}^2$ represents variance component for main effects while $\sigma_{i2}^2$ represents variance component for interaction effects. 

