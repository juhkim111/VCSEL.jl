module UnivariateTest

using Random, LinearAlgebra, Test, VarianceComponentSelect, PenaltyFunctions

Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 10    # no. variance components
p = 3     # no. covariates
X = randn(n, p)
β = ones(p)

V  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] = Vi * Vi'
  V[i] = V[i] ./ norm(V[i])
end
V[m + 1] = Matrix(I, n, n) ./ √n

# truth 
σ2 = zeros(m + 1)
σ2[1] = σ2[4] = σ2[9] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
  global Ω += σ2[i] * V[i]
end
Ωchol = cholesky(Ω)
y = X * β + Ωchol.L * randn(n)

@info "variance component selection with lasso penalty"
σ2path, objpath, λpath = vcselect(y, X, V; penfun=L1Penalty()) 
@test σ2path .>= 0 

@info "variance component selection with no penalty"
temp, = vcselect(y, X, V)
@test temp .>= 0 

@info "variance component selection with adaptive lasso penalty"
penwt = zeros(m + 1)
penwt[1:m] = 1 ./ sqrt.(temp[1:m])
σ2path, objpath, λpath = vcselect(y, X, V; penfun=L1Penalty(), penwt=penwt)
@test σ2path .>= 0  

@info "variance component selection with MCP penalty"
σ2path, objpath, λpath = vcselect(y, X, V; penfun=MCPPenalty()) 
@test σ2path .>= 0 



end 