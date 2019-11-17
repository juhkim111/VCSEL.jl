module standardize_test


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
W  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  #V[i] = Vi * Vi'
  V[i] = Vi * Vi'
  W[i] = V[i] / norm(V[i])
end
V[m + 1] = Matrix(I, n, n) 
W[m + 1] = Matrix(I, n, n) ./ √n

# truth 
σ2 = zeros(m + 1)
σ2[1] = σ2[4] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
ΩW = zeros(n, n)
for i = 1:(m + 1)
   #Ω .+= σ2[i] * V[i]
   axpy!(σ2[i], V[i], Ω)
   axpy!(σ2[i], W[i], ΩW)
end
Ωchol = cholesky(Symmetric(Ω))
yreml = Ωchol.L * randn(n)
y = X * β + Ωchol.L * randn(n)

ΩWchol = cholesky(Symmetric(ΩW))
yWreml = ΩWchol.L * randn(n)
yW = X * β + ΩWchol.L * randn(n)
nlambda = 20 

# # W 
# vcm1 = VCModel(yWreml, W, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
# vcm2 = VCModel(yW, X, W, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

# vcm_nostan = deepcopy(vcm1)
# vcselect!(vcm_nostan; standardize=false)

# vcm_stan = deepcopy(vcm1)
# vcselect!(vcm_stan)

# println("vcm_nostan.Σ = ", vcm_nostan.Σ)
# println("vcm_stan.Σ = ", vcm_stan.Σ)


# V 
vcmV1 = VCModel(yreml, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
vcmV2 = VCModel(y, X, V, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

vcmV_nostan = deepcopy(vcmV2)
vcselect!(vcmV_nostan; standardize=false)

vcmV_stan = deepcopy(vcmV2)
vcselect!(vcmV_stan)

println("vcmV_nostan.Σ = ", vcmV_nostan.Σ)
println("vcmV_stan.Σ = ", vcmV_stan.Σ)

end # end of module 