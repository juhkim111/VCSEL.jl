module VCModel_test

using Random, LinearAlgebra, Test
include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

Random.seed!(123)
# n, m, p = 100, 5, 3
# X = randn(n, p)
# β = ones(p)
# V  = Array{Matrix{Float64}}(undef, m + 1)
# for i = 1:m
#   Vi = randn(n, 50)
#   #V[i] = Vi * Vi'
#   V[i] = Vi * Vi'
#   V[i] = V[i] ./ norm(V[i])
# end
# V[m + 1] = Matrix(I, n, n) ./ √n

# # truth 
# σ2 = zeros(m + 1)
# σ2[1] = σ2[3] = 5.0
# σ2[end] = 1.0

# # form Ω
# Ω = zeros(n, n)
# for i = 1:(m + 1)
#    #Ω .+= σ2[i] * V[i]
#    axpy!(σ2[i], V[i], Ω)
# end
# Ωchol = cholesky(Symmetric(Ω))
# y1 = Ωchol.L * randn(n)
# y2 = X * β + Ωchol.L * randn(n)

# #vcm1 = VCModel(y1, V, [0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
# vcm2 = VCModel(y2, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

# vcselect!(vcm2; penfun=L1Penalty(), λ=5.0)
# println("vcm2.Σ = ", vcm2.Σ)
# println("vcm2.β = ", vcm2.β)

# σ2_est, beta_est, = vcselect(y2, X, V; penfun=L1Penalty(), λ=5.0, σ2=[0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
# println("σ2_est = ", σ2_est)
# println("beta_est =", beta_est)
# @testset begin
#   @test vcm1.Σ == σ2_est
# end 

# vcselect!(vcm2; penfun=L1Penalty(), λ=5.0)
# println(vcm2.Σ)
# σ2,= vcselect(y2, X, V; penfun=L1Penalty(), λ=5.0, σ2=[0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
# println(σ2)
# @testset begin
#   @test vcm2.Σ == σ2
# end 

# @testset begin 
#   @test nvarcomps(vcm1) == m 
#   @test nvarcomps(vcm2) == m
# end 

## generate data from a d-variate (d>1) response variance component model
Random.seed!(123)
n = 100         # no. observations
d = 5           # no. categories
nvarcomps = 6   # no. variance components
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
  Vi = randn(n, 50)
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

# define VCModel
Σinit = [ones(d, d) for i in 1:nvarcomps]
Σinit[end] = Matrix(I, d, d)
#vcmreml = VCModel(Yreml, V, Σinit)
vcm = VCModel(Y, X, V, Σinit)

vcselect!(vcm; penfun=L1Penalty(), λ=25.0)
Σinit = [ones(d, d) for i in 1:nvarcomps]
Σinit[end] = Matrix(I, d, d)
Σ̂, β, = vcselect(Y, X, V; penfun=L1Penalty(), λ=25.0, Σ=Σinit)
println("vcm.Σ = ", vcm.Σ)
println("Σ̂ = ", Σ̂)
println("vcm.β = ", vcm.β)
println("β =", β)
# @testset begin
# @test Σ̂ == vcmreml.Σ
# end

end 