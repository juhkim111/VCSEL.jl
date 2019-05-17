module OperationsTest

using Random, LinearAlgebra, Test  #, VarianceComponentSelect

include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

Random.seed!(123)

## univariate ## 
# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 10    # no. variance components
p = 3     # no. covariates
X = randn(n, p)
β = ones(p)

# generate vector of covariance matrix 
V  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] = Vi * Vi'
end
V[m + 1] = Matrix(I, n, n)

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

@testset "projection onto null space of X" begin
  ynew, Vnew, B = nullprojection(y, X, V)
  for i in 1:(m + 1)
      @test norm(Vnew[i]) ≈ 1
  end
  @test B'B ≈ I 
  @test isapprox(maximum(abs.(B'*X)), 0; atol=1e-8) #all(B'*X .≈ 0)
end 

# @testset "estimate fixed effects" begin 
#   @test betaestimate(y, X, Ω) == betaestimate(y, X, Ωchol)
# end 

## multivariate ## 
# generate data from a d-variate response variance component model
n = 100       # no. observations
d = 3         # no. categories
p = 4         # no. covariates
nvarcomps = 5 # no. variance components 
X = randn(n, p)
β = ones(p, d)

# variance component matrices 
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [1, 3, 5]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end

# vector of covariance matrices; last one is identity matrix
V  = Array{Matrix{Float64}}(undef, nvarcomps)
for i = 1:(nvarcomps - 1)
  Vi = randn(n, 50)
  V[i] =  Vi * Vi'
end
V[end] = Matrix(I, n, n)

# form Ω
Ω = zeros(n*d, n*d)
for i = 1:nvarcomps
    Ω .+= kron(Σ[i], V[i])
end

Ωnew = zeros(n*d, n*d)
for i = 1:nvarcomps
    kronaxpy!(Σ[i], V[i], Ωnew)
end

@testset "kronecker product update" begin
    @test Ωnew ≈ Ω
end

# generate response vector 
Ωchol = cholesky!(Symmetric(Ω))
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)
Ynew, Vnew, B = nullprojection(Y, X, V)
@testset "projection onto null space of X" begin
  for i in 1:nvarcomps
      @test norm(Vnew[i]) ≈ 1
  end
  @test B'B ≈ I 
  @test isapprox(maximum(abs.(B'*X)), 0; atol=1e-8) #all(B'*X .≈ 0)
end 

# call `vcselect` to get REML estimates
Σ̂, obj, niters, = vcselect(Ynew, Vnew; penfun=L1Penalty(), λ=15.0)
Ω = zeros(n*d, n*d)
for j in eachindex(Σ̂)
    if Σ̂[j] == zeros(d, d)
        continue 
    end 
    kronaxpy!(Σ̂[j], V[j], Ω)
end
β̂ = fixedeffects(Y, X, Ω)
β̂2 = fixedeffects(Y, X, V, Σ̂)
@testset "estimate fixed effects parameter" begin 
   @test size(β̂) == size(β)
   @test β̂ == β̂2
end 



end 