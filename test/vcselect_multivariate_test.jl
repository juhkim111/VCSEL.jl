module MultivariateTest 

#using Random, LinearAlgebra, Test, VarianceComponentSelect
include("../src/VarianceComponentSelect.jl") #
using .VarianceComponentSelect #
using Random, LinearAlgebra, Test #

Random.seed!(123)


# generate data from a d-variate response variance component model
n = 100       # no. observations
d = 3       # no. categories
nvarcomps = 6 # no. variance components
# p = 4         # no. covariates
# X = randn(n, p)
# B = ones(p, d)
#Σ = Array{Matrix{Float64}}(undef, nvarcomps)
Σ = [zeros(d, d) for i in 1:nvarcomps]
for i in [2, 6]
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
#Y = X * B + reshape(Ωchol.L * randn(n*d), n, d)
Y = reshape(Ωchol.L * randn(n*d), n, d)
Σ̂, _, _, _, objvec= vcselect(Y, V; verbose=true)

@info "check if objective values are monotonically decreasing (no penalty)" 
@testset begin 
    for i in 1:(length(objvec) - 1)
      @test objvec[i] >= objvec[i+1]
    end 
  end 

Σ̂, _, _, _, objvec = vcselect(Y, V; penfun=L1Penalty(), λ=500.0, verbose=true)

@info "check if objective values are monotonically decreasing (L1 penalty)" 
@testset begin 
    for i in 1:(length(objvec) - 1)
      @test objvec[i] >= objvec[i+1]
    end 
  end 



end 