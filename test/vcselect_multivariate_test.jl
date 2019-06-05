module MultivariateTest 

#using Random, LinearAlgebra, Test, VarianceComponentSelect
include("../src/VarianceComponentSelect.jl") #
using .VarianceComponentSelect #
using Random, LinearAlgebra, Test, DelimitedFiles #

Random.seed!(123)


## generate data from a 1-variate response variance component model
n = 100         # no. observations
d = 1           # no. categories
nvarcomps = 6   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [1, 4, 6]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end
println(Σ)

# vector of covariance matrix 
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
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Y = reshape(Ωchol.L * randn(n*d), n, d)

# make sure estimates from univariate and mutivariate (d=1) algorithm match 
Σ̂, _, _, _, objvec1 = vcselect(Y, V; penfun=L1Penalty(), λ=1.2, verbose=true)
σ̂2, _, _, _, objvec2 = vcselect(Y[:], V; penfun=L1Penalty(), λ=1.2, verbose=true)
@info "check if univariate and multivariate (d=1) match" 
@testset begin 
    @test isapprox(objvec1, objvec2)
    for j in 1:nvarcomps 
      @test isapprox(Σ̂[j][1,1], σ̂2[j])
    end 
  end 


## generate data from a d-variate (d>1) response variance component model
n = 100         # no. observations
d = 4           # no. categories
nvarcomps = 6   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [1, 4, 6]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end

# vector of covariance matrix 
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
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Y = reshape(Ωchol.L * randn(n*d), n, d)

# obtain REML estimates 
Σ̂, = vcselect(Y, V) #; verbose=true)
Σ̂, _, _, _, objvec = vcselect(Y, V; verbose=true)
# test if objective values are monotonically decreasing
@info "check if objective values are monotonically decreasing (no penalty)" 
@testset begin 
    for i in 1:(length(objvec) - 1)
      @test objvec[i] >= objvec[i+1]
    end 
  end 

# obtain REML estimates 
Σ̂, _, _, _, objvec = vcselect(Y, V; penfun=L1Penalty(), λ=1.2, verbose=true)
# test if objective values are monotonically decreasing
@info "check if objective values are monotonically decreasing (L1 penalty)" 
@testset begin 
    for i in 1:(length(objvec) - 1)
      @test objvec[i] >= objvec[i+1]
    end 
  end 

# generate response vector (covariate matrix)
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)
Σ̂, β̂, _, _, _, objvec = vcselect(Y, X, V; penfun=L1Penalty(), λ=25.0, verbose=true)

# test if dimension matches 
@info "check if dimension of fixed effects parameter matches"
@testset begin 
  @test size(Σ̂[1]) == size(Σ[1])
end 

# test if objective values are monotonically decreasing
@info "check if objective values are monotonically decreasing (L1 penalty)" 
@testset begin 
    for i in 1:(length(objvec) - 1)
      @test objvec[i] >= objvec[i+1]
    end 
  end 

# using Profile 
# Profile.clear()
# nlambda = 20 
# Profile.init(n=10^7, delay=0.01)
# @profile Σ̂path, = vcselectpath(Y, V; penfun=L1Penalty(), nlambda=nlambda)
#Profile.print(format=:flat)

end 


