module UtilitiesTest

<<<<<<< HEAD
# load packages
using Random, LinearAlgebra, Test, VCSEL

# set seed 
=======
#using Random, LinearAlgebra, Test, VCSEL
using Random, LinearAlgebra, Test
include("../src/VCSEL.jl")
using .VCSEL

>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
Random.seed!(123)

# # generate data from an univariate response variance component model 
# n = 100   # no. observations
# m = 10    # no. variance components
# p = 3     # no. covariates
# X = randn(n, p)
# β = ones(p)

# # generate vector of covariance matrix 
# V  = Array{Matrix{Float64}}(undef, m + 1)
# for i = 1:m
#   Vi = randn(n, 50)
#   V[i] = Vi * Vi'
# end
# V[m + 1] = Matrix(I, n, n)

# # make sure frobenius norm equals to 1 
# checkfrobnorm!(V)
# @testset "check frob norm 1" begin 
#   for Vi in V 
#     @test norm(Vi) ≈ 1
#   end 
# end 

# # truth 
# σ2 = zeros(m + 1)
# σ2[1] = σ2[4] = σ2[9] = 5.0
# σ2[end] = 1.0

# # form Ω
# Ω = zeros(n, n)
# for i = 1:(m + 1)
#   global Ω += σ2[i] * V[i]
# end
# Ωchol = cholesky(Ω)
# y = X * β + Ωchol.L * randn(n)
# vcm = VCModel(y, X, V, [fill(0.5, m); 1.0])

# # @testset "projection onto null space of X" begin
# #   ynew, Vnew, B = nullprojection(y, X, V)
# #   for i in 1:(m + 1)
# #       @test norm(Vnew[i]) ≈ 1
# #   end
# #   @test B'B ≈ I 
# #   @test isapprox(maximum(abs.(B'*X)), 0; atol=1e-8) #all(B'*X .≈ 0)
# # end 

# # @testset "estimate fixed effects" begin 
# #   @test betaestimate(y, X, Ω) == betaestimate(y, X, Ωchol)
# # end 

# Σ̂path, β̂path, λpath, _, niterspath = vcselectpath!(vcm; penfun=L1Penalty(), nλ=20)
# println()
# println("nrows of Σ̂path = ", size(Σ̂path, 1))
# println("ncols of Σ̂path = ", size(Σ̂path, 2))

#### multivariate ####
## generate data from a d-variate (d>1) response variance component model
Random.seed!(123)
n = 50         # no. observations
d = 3           # no. categories
nvarcomps = 5   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [1, 4]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end
Σ[end] = Matrix(I, d, d)

# vector of covariance matrix 
V  = Array{Matrix{Float64}}(undef, nvarcomps)
for i = 1:(nvarcomps - 1)
  Vi = randn(n, 20)
  V[i] = Vi * Vi'
  V[i] = V[i] ./ norm(V[i])
end
V[end] = Matrix(I, n, n) ./ √n

# form Ω
Ω = zeros(n*d, n*d)
for i = 1:nvarcomps
    Ω .+= kron(Σ[i], V[i])
end
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Yreml = reshape(Ωchol.L * randn(n*d), n, d)
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)

Σinit = [ones(d, d) for i in 1:nvarcomps]
Σinit[end] = Matrix(I, d, d)
vcm = VCModel(Y, X, V, Σinit)
nlambda = 10
Σ̂path, betapath, lambdapath, objpath,  = vcselectpath!(vcm; penfun=L1Penalty(), 
   nλ=nlambda)

ranking, rest = rankvarcomps(Σ̂path)

@testset "rankvarcomps test" begin
for i in ranking 
   tmp = findall(x -> norm(x) > 1e-6, view(Σ̂path, i, 2:nlambda))
   @test !isempty(tmp)
end 
end 

end # end of module 