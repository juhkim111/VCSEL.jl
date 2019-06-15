module VCModel_test

using Random, LinearAlgebra, Test
include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

n, m, p = 100, 5, 3
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
   #Ω .+= σ2[i] * V[i]
   axpy!(σ2[i], V[i], Ω)
end
Ωchol = cholesky(Symmetric(Ω))
y1 = Ωchol.L * randn(n)
y2 = X * β + Ωchol.L * randn(n)

vcm1 = VCModel(y1, V, [0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcm2 = VCModel(y2, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

@testset begin 
  @test nvarcomps(vcm1) == m 
  @test nvarcomps(vcm2) == m
end 




end 