module UnivariateTest

<<<<<<< HEAD
# load packages 
using Random, LinearAlgebra, VCSEL, Test
=======
#using Random, LinearAlgebra, VarianceComponentSelect, Test
include("../src/VCSEL.jl")
using Random, LinearAlgebra, .VCSEL, Test
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

# set seed 
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
  V[i] = Vi * Vi'
  V[i] = V[i] ./ norm(V[i])
end
V[end] = Matrix(I, n, n) ./ √n

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

<<<<<<< HEAD
@info "check if objective values are monotonically decreasing" 
σ2, _, _, _, _, objvec = vcselect(y, X, V; λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing" 
σ2, _, _, _, objvec = vcselect(yreml, V; λ=2.0, verbose=true)
=======
vcm1 = VCModel(yreml, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcm2 = VCModel(y, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])


@info "objective values are monotonically decreasing (no penalty)" 
vcm11 = deepcopy(vcm1)
_, _, _, objvec = vcselect!(vcm11; verbose=true)
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 
temp1 = vcm11.Σ

<<<<<<< HEAD
## variance component selection at specific lambda 
σ2_tmp, = vcselect(yreml, V)

@info "solution path with no penalty (REML)"
σ2path, = vcselectpath(yreml, V)
@testset begin
  @test σ2_tmp == σ2path
=======
vcm21 = deepcopy(vcm2)
_, _, _, objvec = vcselect!(vcm21; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488
end 
temp2 = vcm21.Σ

<<<<<<< HEAD
@info "solution path with no penalty"
temp, = vcselectpath(y, X, V)
@test all(temp .>= 0)

@info "lasso penalty at λ = 2.0 "
σ2, β, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0)
@test all(σ2 .>= 0)
=======
@info "objective values are monotonically decreasing (L1 Penalty)"
vcm22 = deepcopy(vcm2)
_, _, _, objvec = vcselect!(vcm22; 
      penfun=L1Penalty(), λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

@info "check if objective values are monotonically decreasing (adaptive L1 Penalty)" 
vcm12 = deepcopy(vcm1)
penwt = zeros(m + 1)
<<<<<<< HEAD
penwt[1:m] = 1 ./ sqrt.(temp[1:m])
σ2, β, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0, penwt=penwt)
@test all(σ2 .>= 0)

@info "MCP penalty at λ = 2.0"
σ2, β, obj, = vcselect(y, X, V; penfun=MCPPenalty(), λ=2.0)
@test all(σ2 .>= 0)
=======
penwt[1:m] = 1 ./ sqrt.(temp1[1:m])
_, _, _, objvec = vcselect!(vcm12; penwt=penwt, verbose=true, λ=5.0)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (adaptive L1 penalty)" 
vcm22 = deepcopy(vcm2)
penwt = zeros(m + 1)
penwt[1:m] = 1 ./ sqrt.(temp2[1:m])
_, _, _, objvec = vcselect(y, V; penfun=L1Penalty(), λ=15.0, verbose=true,
      penwt = penwt)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

@info "check if objective values are monotonically decreasing (MCP penalty)" 
vcm12 = deepcopy(vcm1)
_, _, _, objvec = vcselect!(vcm12; penfun=MCPPenalty(), λ=5.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

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
      
<<<<<<< HEAD
    @info "solution path with lasso penalty (REML)"
    σ2path, = vcselectpath(yreml, V; penfun=L1Penalty(), nlambda=nlambda)
    @test all(σ2path .>= 0)

    @info "solution path with adaptive lasso penalty (REML)"
    σ2path, = vcselectpath(yreml, V; penfun=L1Penalty(), penwt=penwt, nlambda=nlambda)
    @test all(σ2path .>= 0)

    @info "solution path with MCP penalty (REML)"
    σ2path, = vcselectpath(yreml, V; penfun=MCPPenalty(), penwt=penwt, nlambda=nlambda) 
    @test all(σ2path .>= 0)

    @info "solution path with lasso penalty"
    σ2path, objpath, λpath, = vcselectpath(y, X, V; penfun=L1Penalty(), nlambda=nlambda) 
    @test all(σ2path .>= 0)

    @info "solution path with adaptive lasso penalty"
    σ2path, objpath, λpath, = vcselectpath(y, X, V; 
        penfun=L1Penalty(), penwt=penwt, nlambda=nlambda) 
    @test all(σ2path .>= 0)

    @info "solution path with MCP penalty"
    σ2path, objpath, λpath, = vcselectpath(y, X, V; 
        penfun=MCPPenalty(), penwt=penwt, nlambda=nlambda) 
    @test all(σ2path .>= 0)
=======
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
>>>>>>> 03d88815ca569114a4903b807fc6d669e3c38488

# end 


end 