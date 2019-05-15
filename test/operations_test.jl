module OperationsTest



using Random, LinearAlgebra, Test  #, VarianceComponentSelect

include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

Random.seed!(123)

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

#
d = 3         # no. categories
nvarcomps = 5 
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [1, 3, 5]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end
## make the first variance component 0 matrix
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



end 