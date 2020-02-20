module MaxLambdaTest

using Random, LinearAlgebra, VCSEL, Test

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
   Ω .+= σ2[i] * V[i]
end
Ωchol = cholesky(Symmetric(Ω))
y = Ωchol.L * randn(n)

maxλ_lasso = maxlambda(y, V; penfun=L1Penalty())
maxλ_mcp = maxlambda(y, V; penfun=MCPPenalty())
σ2_lasso, = vcselect(y, V; penfun=L1Penalty(), λ=maxλ_lasso) 
σ2_mcp, = vcselect(y, V; penfun=MCPPenalty(), λ=maxλ_mcp) 

tol = 1e-6
@testset  begin 
    @test all(σ2_lasso[:, 1:end-1] .> tol)
    @test all(σ2_mcp[:, 1:end-1] .> tol)
end 


end 