module UnivariateTest

# load packages 
using Random, LinearAlgebra, VarianceComponentSelect, Test

# set seed 
Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 10    # no. variance components
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
σ2[1] = σ2[4] = σ2[9] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
   Ω .+= σ2[i] * V[i]
end
Ωchol = cholesky(Symmetric(Ω))
yreml = Ωchol.L * randn(n)
y = X * β + Ωchol.L * randn(n)
nlambda = 20 

@info "check if objective values are monotonically decreasing" 
σ2, _, _, _, _, objvec = vcselect(y, X, V; λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing" 
σ2, _, _, _, objvec = vcselect(yreml, V; λ=2.0, verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

## variance component selection at specific lambda 
σ2_tmp, = vcselect(yreml, V)

@info "solution path with no penalty (REML)"
σ2path, = vcselectpath(yreml, V)
@testset begin
  @test σ2_tmp == σ2path
end 

@info "solution path with no penalty"
temp, = vcselectpath(y, X, V)
@test all(temp .>= 0)

@info "lasso penalty at λ = 2.0 "
σ2, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0)
@test all(σ2 .>= 0)

@info "adaptive lasso penalty at λ = 2.0"
penwt = zeros(m + 1)
penwt[1:m] = 1 ./ sqrt.(temp[1:m])
σ2, obj, = vcselect(y, X, V; penfun=L1Penalty(), λ=2.0, penwt=penwt)
@test all(σ2 .>= 0)

@info "MCP penalty at λ = 2.0"
σ2, obj, = vcselect(y, X, V; penfun=MCPPenalty(), λ=2.0)
@test all(σ2 .>= 0)

## solution path 
@testset begin 
      
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

end 


end 