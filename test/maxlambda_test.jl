module MaxLambdaTest

#using Random, LinearAlgebra, VarianceComponentSelect, Test
include("../src/VarianceComponentSelect.jl") #
using .VarianceComponentSelect #
using Random, LinearAlgebra, Test #
Random.seed!(123)
tol = 1e-10

Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 5     # no. variance components
p = 4     # no. covariates
X = randn(n, p)
β = ones(p)

V  = Array{Matrix{Float64}}(undef, m + 1)
W  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] = Vi * Vi'
  W[i] = V[i] ./ norm(V[i])
end
V[end] = Matrix(I, n, n) 
W[end] = Matrix(I, n, n) ./ √n

# truth 
σ2 = zeros(m + 1)
σ2[1] = σ2[4] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
   axpy!(σ2[i], W[i], Ω)
end
Ωchol = cholesky(Symmetric(Ω))
y = Ωchol.L * randn(n)
Y = reshape(y, n, 1)

# find maximum lambda 
maxλ_lasso, = maxlambda(y, V; penfun=L1Penalty())
maxλ_mcp, = maxlambda(y, V; penfun=MCPPenalty())
σ̂2_lasso, = vcselect(y, V; penfun=L1Penalty(), λ=maxλ_lasso) 
σ̂2_mcp, = vcselect(y, V; penfun=MCPPenalty(), λ=maxλ_mcp) 

tol = 1e-10
@testset  begin 
    @test all(σ̂2_lasso[:, 1:end-1] .> tol)
    @test all(σ̂2_mcp[:, 1:end-1] .> tol)
end 

maxλ_lasso, = maxlambda(Y, V; penfun=L1Penalty())
σ̂2_lasso, = vcselect(Y, V; penfun=L1Penalty(), λ=maxλ_lasso) 

@testset begin 
  for i in 1:m
    @test isapprox(σ̂2_lasso[i], zeros(1, 1); atol=tol)
  end 
end 

## generate data from a d-variate response variance component model
n = 100         # no. observations
d = 3           # no. categories
m = 5   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:(m+1)]
for i in [1, 4, 6]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end

# vector of covariance matrix 
V  = Array{Matrix{Float64}}(undef, m+1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] =  Vi * Vi'
end
V[end] = Matrix(I, n, n)

# form Ω
Ω = zeros(n*d, n*d)
for i = 1:(m + 1)
    Ω .+= kron(Σ[i], V[i])
end
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Y = reshape(Ωchol.L * randn(n*d), n, d)

# find maximum lambda and test 
maxλ_lasso, iter = maxlambda(Y, V; penfun=L1Penalty())
Σ̂_lasso, = vcselect(Y, V; penfun=L1Penalty(), λ=maxλ_lasso) 

@testset begin 
    for i in 1:m
      @test isapprox(Σ̂_lasso[i], zeros(d, d); atol=tol)
    end 
end 

end 