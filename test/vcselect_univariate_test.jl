module UnivariateTest

#using Random, LinearAlgebra, VarianceComponentSelect, Test
include("../src/VarianceComponentSelect.jl")
using Random, LinearAlgebra, .VarianceComponentSelect, Test

Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 6     # no. variance components
p = 3     # no. covariates
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
σ2[1] = σ2[4] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
   #Ω .+= σ2[i] * V[i]
   axpy!(σ2[i], V[i], Ω)
end
Ωchol = cholesky(Symmetric(Ω))
yreml = Ωchol.L * randn(n)
y = X * β + Ωchol.L * randn(n)
nlambda = 20 

vcm1 = VCModel(yreml, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcm2 = VCModel(y, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcm22 = VCModel(y, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

@info "objective values are monotonically decreasing (no penalty)" 
Σ̂1, β̂1, obj1, niters1, Ω̂1, objvec = vcselect!(vcm1; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

Σ̂2, β̂2, obj2, niters2, Ω̂2, objvec = vcselect!(vcm2; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "objective values are monotonically decreasing (L1 Penalty)"
Σ̂22, β̂22, obj22, niters22, _, objvec22 = vcselect!(vcm22; 
      penfun=L1Penalty(), λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec22) - 1)
    @test objvec22[i] >= objvec22[i+1]
  end 
end 

# reset and test if the same wth the previous 
resetVCModel!(vcm2, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
Σ̂_L1, β̂_L1, obj_L1, niters_L1, _, objvec_L1 = vcselect!(vcm2; 
        penfun=L1Penalty(), λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec_L1) - 1)
    @test objvec_L1[i] >= objvec_L1[i+1]
  end 
end 

@info "check if resetVCModel! works"
@testset begin 
  @test Σ̂22 == Σ̂_L1 
  @test β̂22 == β̂_L1
  @test obj22 == obj_L1
  @test niters22 == niters_L1
end 

# @info "check if objective values are monotonically decreasing (no penalty)" 
# temp, _, _, _, objvec = vcselect(y, V; verbose=true)
# penwt = zeros(m + 1)
# penwt[1:m] = 1 ./ sqrt.(temp[1:m])
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# @info "check if objective values are monotonically decreasing (L1 penalty)" 
# σ2, _, _, _, objvec = vcselect(y, V; penfun=L1Penalty(), λ=15.0, verbose=true,
#       penwt = penwt)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# @info "check if objective values are monotonically decreasing (MCP penalty)" 
# σ2, _, _, _, objvec = vcselect(y, V; penfun=MCPPenalty(), λ=22.0, verbose=true)
# @testset begin 
#   for i in 1:(length(objvec) - 1)
#     @test objvec[i] >= objvec[i+1]
#   end 
# end 

# ## variance component selection at specific lambda 
# σ2_tmp, = vcselect(y, V)

# @info "solution path with no penalty (REML)"
# σ2path, = vcselectpath(y, V)
# @testset begin
#   @test σ2_tmp == σ2path
# end 

# @info "solution path with no penalty"
# temp, = vcselectpath(y, X, V)
# @test all(temp .>= 0)

# @info "lasso penalty at λ = 2.0 "
# σ2, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0)
# @test all(σ2 .>= 0)

# @info "adaptive lasso penalty at λ = 2.0"
# penwt = zeros(m + 1)
# penwt[1:m] = 1 ./ sqrt.(temp[1:m])
# σ2, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0, penwt=penwt)
# @test all(σ2 .>= 0)

# @info "MCP penalty at λ = 2.0"
# σ2, obj, = vcselect(y, X, V; penfun=MCPPenalty(), λ=2.0)
# @test all(σ2 .>= 0)

# ## solution path 
# @testset begin 
      
#     @info "solution path with lasso penalty (REML)"
#     σ2path, = vcselectpath(y, V; penfun=L1Penalty(), nlambda=nlambda)
#     @test all(σ2path .>= 0)

#     @info "solution path with adaptive lasso penalty (REML)"
#     σ2path, = vcselectpath(y, V; penfun=L1Penalty(), penwt=penwt, nlambda=nlambda)
#     @test all(σ2path .>= 0)

#     @info "solution path with MCP penalty (REML)"
#     σ2path, = vcselectpath(y, V; penfun=MCPPenalty(), penwt=penwt, nlambda=nlambda) 
#     @test all(σ2path .>= 0)

#     @info "solution path with lasso penalty"
#     σ2path, objpath, λpath = vcselectpath(y, X, V; penfun=L1Penalty(), nlambda=nlambda) 
#     @test all(σ2path .>= 0)

#     @info "solution path with adaptive lasso penalty"
#     σ2path, objpath, λpath = vcselectpath(y, X, V; 
#         penfun=L1Penalty(), penwt=penwt, nlambda=nlambda) 
#     @test all(σ2path .>= 0)

#     @info "solution path with MCP penalty"
#     σ2path, objpath, λpath = vcselectpath(y, X, V; 
#         penfun=MCPPenalty(), penwt=penwt, nlambda=nlambda) 
#     @test all(σ2path .>= 0)

# end 


end 