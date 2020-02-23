module VCModel_test

using Random, LinearAlgebra, Test
include("../src/VCSEL.jl")
using .VCSEL

Random.seed!(123)
n, m, p = 100, 6, 3
X = randn(n, p)
β = ones(p)
V  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  #V[i] = Vi * Vi'
  V[i] = Vi * Vi'
  V[i] = V[i] ./ norm(V[i])
end
V[m + 1] = Matrix(I, n, n) ./ √n

# truth 
σ2 = zeros(m + 1)
σ2[1] = σ2[3] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
   axpy!(σ2[i], V[i], Ω)
end
Ωchol = cholesky(Symmetric(Ω))
yreml = Ωchol.L * randn(n)
y = X * β + Ωchol.L * randn(n)

vcm1 = VCModel(yreml, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcm2 = VCModel(y, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

@testset begin
  @test length(vcm1) == 1
  @test ncovariates(vcm2) == p 
  @test size(vcm1) == (n, )
  @test nmeanparams(vcm2) == p 
  @test nvarcomps(vcm1) == m + 1
end 

## generate data from a d-variate (d>1) response variance component model
Random.seed!(123)
n = 50         # no. observations
d = 3           # no. categories
m = 5   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:(m+1)]
for i in [1, 4]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end
Σ[end] = Matrix(I, d, d)

# vector of covariance matrix 
V  = Array{Matrix{Float64}}(undef, m+1)
for i = 1:m
  Vi = randn(n, 20)
  V[i] = Vi * Vi'
  V[i] = V[i] ./ norm(V[i])
end
V[end] = Matrix(I, n, n) ./ √n

# form Ω
Ω = zeros(n*d, n*d)
for i = 1:(m + 1)
    Ω .+= kron(Σ[i], V[i])
end
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Yreml = reshape(Ωchol.L * randn(n*d), n, d)
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)

# define VCModel
Σinit = [Matrix(1.0*I, d, d) for i in 1:(m+1)]
vcm1 = VCModel(Yreml, V)
vcm1_init = VCModel(Yreml, V, Σinit)

vcm2 = VCModel(Y, X, V)
vcm2_init = VCModel(Y, X, V, Σinit)

@testset begin 
  @test nvarcomps(vcm1) == m + 1
  @test vcm1.Σ == vcm1_init.Σ
  @test vcm2.Σ == vcm2_init.Σ
  @test vcm1.Ω == vcm1_init.Ω
  @test vcm2.Ω == vcm2_init.Ω
end 


end 