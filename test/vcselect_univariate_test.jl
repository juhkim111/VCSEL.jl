module UnivariateTest

using Random, LinearAlgebra, Test, VarianceComponentSelect, PenaltyFunctions

Random.seed!(123)

n = 100   # no. observations
m = 10    # no. variance components
p = 3     # no. covariates
X = randn(n, p)
β = ones(p)

## make the first variance component 0 matrix
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

# lasso penalty 
println("lasso penalty")
σ2path, objpath, λpath = vcselect(y, X, V; penfun=L1Penalty()) 
println("σ2path=$σ2path")
println("objpath=$objpath")
println("λpath=$λpath")

# adaptive lasso penalty

# mcp penalty 
println("MCP penalty")
σ2path, objpath, λpath = vcselect(y, X, V; penfun=MCPPenalty()) 
println("σ2path=$σ2path")
println("objpath=$objpath")
println("λpath=$λpath")

# no penalty 
println("No penalty")
σ2path, objpath, λpath = vcselect(y, X, V; penfun=NoPenalty()) 
println("σ2path=$σ2path")
println("objpath=$objpath")
println("λpath=$λpath")

end 