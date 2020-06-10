# VCSEL.jl

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juhkim111.github.io/VCSEL.jl/dev)  | [![Build Status](https://travis-ci.org/juhkim111/VCSEL.jl.svg?branch=master)](https://travis-ci.org/juhkim111/VCSEL.jl) | [![Coverage Status](https://coveralls.io/repos/github/juhkim111/VCSEL.jl/badge.svg?branch=master)](https://coveralls.io/juhkim111/VCSEL.jl?branch=master)  

**VCSEL** is a Julia package for implementing [Majorization-Minimization (MM) algorithm](https://en.wikipedia.org/wiki/MM_algorithm) that selects variance components that are associated with response variable(s) via penalization method. 

## Package Features 

**VCSEL** supports variance component selection for

* a univariate response model 
* a univariate response model with interaction terms 
* a multivariate response model 

using lasso, adaptive lasso, or MCP penalty.  

## Installation 

This package requires Julia v0.7.0 or later, which can be obtained from
[https://julialang.org/downloads/](https://julialang.org/downloads/) or by building Julia from the sources in the
[https://github.com/JuliaLang/julia](https://github.com/JuliaLang/julia) repository.

The package has not yet been registered and must be installed using the repository location.
Start julia and use the `]` key to switch to the package manager mode and type the following (after `pkg>`):

```julia
(@v1.4) pkg> add https://github.com/juhkim111/VCSEL.jl
```

Use the backspace key to return to the Julia REPL.

## Citation 
