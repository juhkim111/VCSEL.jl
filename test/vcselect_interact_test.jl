module vcselect_interact_test

include("../src/VarianceComponentSelect.jl")
using Random, LinearAlgebra, StatsBase, .VarianceComponentSelect, Test

Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 10    # no. variance components
p = 3     # no. covariates
X = randn(n, p)
β = ones(p)

G  = Array{Matrix{Float64}}(undef, m)
V  = Array{Matrix{Float64}}(undef, m + 1)
Vint  = Array{Matrix{Float64}}(undef, m)
trt = zeros(Int, n)
sample!([0, 1], trt)
trtmat = Diagonal(trt)
for i = 1:m
  G[i] = randn(n, 50)
  V[i] = G[i] * G[i]'
  Vint[i] = trtmat * V[i] * trtmat 
#   V[i] ./= norm(V[i])
#   Vint[i] ./= norm(Vint[i])
end
V[end] = Matrix(I, n, n) #./ √n

# truth 
σ2, σ2int = zeros(m + 1), zeros(m)
σ2[1] = σ2[4] = σ2[9] = 5.0
σ2int[1] = σ2int[4] = σ2int[9] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:m
   Ω .+= σ2[i] * V[i]
   Ω .+= σ2int[i] * Vint[i]
end
Ω .+= σ2[end] * V[end]

Ωchol = cholesky(Symmetric(Ω))
y = Ωchol.L * randn(n)
y2 = X * β + Ωchol.L * randn(n)
nlambda = 20 

# initialize VCintModel 
vcm = VCintModel(y, V, Vint)
vcmX = VCintModel(y, X, V, Vint)

# vcselect with no penalty 
@info "check if objective values are monotonically decreasing (no penalty)"
_, obj, niters, objvec = vcselect!(vcm; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (no penalty)"
_, obj, niters, objvec = vcselect!(vcmX; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end

# reset model 
resetModel!(vcm)
resetModel!(vcmX)

# vcselect with penalty 
@info "check if objective values are monotonically decreasing (L1 penalty)"
_, obj, niters, objvec = vcselect!(vcm; penfun=L1Penalty(), λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (L1 penalty)"
_, obj, niters, objvec = vcselect!(vcmX; penfun=L1Penalty(), λ=1.5, verbose=true,
                                        standardize=false)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end

println("vcm.Σ=", vcm.Σ)
println("vcm.Σint=", vcm.Σint)
println("vcm.β=", vcm.β)
println("vcmX.Σ=", vcmX.Σ)
println("vcmX.Σint=", vcmX.Σint)
println("vcmX.β=", vcmX.β)

# obtains solution path 

end # end of module 