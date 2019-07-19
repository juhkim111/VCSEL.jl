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
  V[i] ./= norm(V[i])
  Vint[i] ./= norm(Vint[i])
end
V[end] = Matrix(I, n, n) ./ √n

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

@info "check if objective values are monotonically decreasing"
σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, G, trt; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, G, trt; verbose=true, λ=2.0, 
      penfun=L1Penalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, β̂, obj, niters, Ω, objvec = vcselect(y2, X, G, trtmat; verbose=true, λ=2.0, 
      penfun=L1Penalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "test path function: no penalty"
σ2path_nopen, σ2intpath_nopen, objpath_nopen, λpath_nopen, niterspath_nopen = 
      vcselectpath(y, G, trt)

@info "test path function: L1Penalty"
λpath = range(0, 4.0, length=5)
σ2path, σ2intpath, objpath, λpath, niterspath = vcselectpath(y, G, trt; penfun=L1Penalty(), 
      λpath=λpath)

@testset begin 
  @test σ2path_nopen == σ2path[:, 1]
  @test σ2intpath_nopen == σ2intpath[:, 1]
  @test objpath_nopen == objpath[1]
  @test niterspath_nopen == niterspath[1]
  @test isempty(λpath_nopen)

end 

##########
# σ̂2_1, σ̂2_2, obj, niters, Ω, objvec = vcselect(y_nocov, V1, V2; penfun=L1Penalty(), 
#         λ=1.2, verbose=true)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# σ̂2_1, σ̂2_2, obj, niters, Ω, objvec = vcselect(y_nocov, V1, V2; penfun=L1Penalty(), 
#         λ=2.0, verbose=true)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# σ̂2_1, σ̂2_2, β, obj, niters, Ω, objvec = vcselect(y, X, V1, V2; verbose=true)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# σ̂2_1, σ̂2_2, β, obj, niters, Ω, objvec = vcselect(y, X, V1, V2; penfun=L1Penalty(), 
#         λ=1.2, verbose=true)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 



end 