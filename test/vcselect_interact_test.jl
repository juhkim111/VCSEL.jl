module InteractTest

# load packages 
using Random, LinearAlgebra, Test, StatsBase # , VarianceComponentSelect
include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

# set seed 
Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 10    # no. variance components
p = 3     # no. covariates
X = randn(n, p)
β = ones(p)

V1  = Array{Matrix{Float64}}(undef, m + 1)
tmp = zeros(Int, n)
V2  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  sample!([0, 1], tmp)
  V1[i] = Vi * Vi'
  V2[i] = Diagonal(tmp) * V1[i] * Diagonal(tmp)
  V1[i] = V1[i] ./ norm(V1[i])
  V2[i] = V2[i] ./ norm(V2[i])
end
V1[end] = Matrix(I, n, n) ./ √n
V2[end] = Matrix(I, n, n) ./ √n

# truth 
σ2_1, σ2_2 = zeros(m + 1), zeros(m + 1)
σ2_1[1] = σ2_1[4] = σ2_1[9] = 5.0
σ2_2[1] = σ2_2[4] = σ2_2[9] = 5.0
σ2_1[end] = σ2_2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:m
   Ω .+= σ2_1[i] * V1[i]
   Ω .+= σ2_2[i] * V2[i]
end
Ω .+= σ2_1[end] * V1[end]

Ωchol = cholesky(Symmetric(Ω))
yreml = Ωchol.L * randn(n)
y = X * β + Ωchol.L * randn(n)
nlambda = 20 

@info "check if objective values are monotonically decreasing"
σ̂2_1, σ̂2_2, obj, niters, Ω, objvec = vcselect(yreml, V1, V2; penfun=L1Penalty(), 
        λ=1.2, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2_1, σ̂2_2, obj, niters, Ω, objvec = vcselect(yreml, V1, V2; penfun=L1Penalty(), 
        λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 



end 