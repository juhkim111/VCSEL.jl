module vcselectTest

#using Random, LinearAlgebra, VarianceComponentSelect, Test
include("../src/VarianceComponentSelect.jl")
using Random, LinearAlgebra, .VarianceComponentSelect, Test

Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 5     # no. variance components
p = 4     # no. covariates
X = randn(n, p)
β = ones(p)

V  = Array{Matrix{Float64}}(undef, m + 1)
W  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] = Vi * Vi'
  W[i] = V[i] ./ norm(V[i])
end
V[end] = Matrix(I, n, n) 
W[end] = Matrix(I, n, n) ./ √n

# truth 
σ2 = zeros(m + 1)
σ2[1] = σ2[4] = 5.0
σ2[end] = 1.0

# form Ω
Ω = zeros(n, n)
for i = 1:(m + 1)
   #Ω .+= σ2[i] * V[i]
   axpy!(σ2[i], W[i], Ω)
end
Ωchol = cholesky(Symmetric(Ω))
y2 = Ωchol.L * randn(n)
Y2 = reshape(y2, n, 1)
y = X * β + Ωchol.L * randn(n)
Y = reshape(y, n, 1)
nlambda = 20 

vcm_y2 = VCModel(y2, V, ones(m + 1))
vcm1_y2 = VCModel(y2, V)

vcm_Y2 = VCModel(Y2, V, [ones(1, 1) for i in 1:m+1])
vcm1_Y2 = VCModel(Y2, V)

vcm1_y = VCModel(y, X, V)
vcm_y = VCModel(y, X, V, ones(m+1))

vcm1_Y = VCModel(Y, X, V)
vcm_Y = VCModel(Y, X, V, [ones(1, 1) for i in 1:m+1])

_, obj1, niters1 = vcselect!(vcm_y2; penfun=L1Penalty(), λ=2.5)
_, obj2, niters2  = vcselect!(vcm1_y2; penfun=L1Penalty(), λ=2.5)

@testset begin 
  @test obj1 == obj2 
  @test niters1 == niters2
  @test vcm_y2.Σ == vcm1_y2.Σ
end 

_, obj1, niters1  = vcselect!(vcm_Y2; verbose=true, standardize=false, 
    penfun=L1Penalty(), λ=2.5)
_, obj2, niters2 = vcselect!(vcm1_Y2; verbose=true, standardize=false, 
    penfun=L1Penalty(), λ=2.5)

@testset begin 
  @test obj1 == obj2 
  @test niters1 == niters2
  @test vcm_Y2.Σ == vcm1_Y2.Σ
end 

resetVCModel!(vcm_Y2)

vcselect!(vcm_Y2; penfun=L1Penalty(), λ=2.5)

@testset for i in eachindex(vcm_Y2.Σ)
  @test vcm_y2.Σ[i] ≈ vcm_Y2.Σ[i][1]
end

_, obj1, niters1 = vcselect!(vcm1_y; penfun=MCPPenalty())
_, obj2, niters2 = vcselect!(vcm_y; penfun=MCPPenalty(), λ=1.0)
prevΣ = deepcopy(vcm_y.Σ)
prevβ = deepcopy(vcm_y.β)

@testset begin 
  @test obj1 == obj2 
  @test niters1 == niters2
  @test vcm1_y.Σ == vcm_y.Σ
  @test vcm1_y.β == vcm_y.β
end 

resetVCModel!(vcm_y)
_, obj22, niters22 = vcselect!(vcm_y; penfun=MCPPenalty())
@testset begin
  @test obj2 == obj22
  @test niters2 == niters22
  @test vcm_y.Σ == prevΣ
  @test vcm_y.β == prevβ
end 

_, obj1, niters1 = vcselect!(vcm1_Y; penfun=NoPenalty(), λ=1.0)
_, obj2, niters2 = vcselect!(vcm_Y)

@testset begin 
  @test obj1 == obj2 
  @test niters1 == niters2
  @test vcm1_Y.β == vcm_Y.β
  @test vcm1_Y.Σ == vcm_Y.Σ
end 

resetVCModel!(vcm_y)
resetVCModel!(vcm_y2)
Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm_y; 
      penfun=MCPPenalty(), λpath=range(0, 8, length=10))
ranking, = rankvarcomps(Σ̂path)

Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm_y2; 
      penfun=L1Penalty(), λpath=range(0, 6, length=5))
ranking, = rankvarcomps(Σ̂path)


resetVCModel!(vcm_Y2)
Σ̂path2, β̂path2, λpath2, objpath2, niterspath2 = vcselectpath!(vcm_Y2; 
      penfun=L1Penalty(), λpath=range(0, 6, length=5))
ranking2, = rankvarcomps(Σ̂path2)

@testset for j in 1:size(Σ̂path2, 2)
  for i in 1:nvarcomps(vcm_y2)
     @test Σ̂path[i, j] ≈ Σ̂path2[i, j][1,1]
  end 
end 
@testset begin
  @test niterspath == niterspath2
  @test ranking == ranking2
end 

resetVCModel!(vcm_Y, [Matrix(1.0*I, 1, 1) for i in 1:nvarcomps(vcm_Y)])

## generate data from a 3-variate response variance component model
Random.seed!(123)
n = 100         # no. observations
d = 3           # no. categories
m = 5   # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 

# variance component matrix 
Σ = [zeros(d, d) for i in 1:(m + 1)]
for i in [1, 4]
  Σi = randn(d, d)
  Σ[i] = Σi * Σi'
end
Σ[end] = Matrix(1.0*I, d, d)

V  = Array{Matrix{Float64}}(undef, m + 1)
for i = 1:m
  Vi = randn(n, 50)
  V[i] = Vi * Vi'
end
V[end] = Matrix(1.0*I, n, n) 

# form Ω
Ω = zeros(n*d, n*d)
for i = 1:(m + 1)
    Ω .+= kron(Σ[i], V[i])
end
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)
Y2 = reshape(Ωchol.L * randn(n*d), n, d)

vcm = VCModel(Y, X, V)
vcm1 = VCModel(Y, X, V, [Matrix(1.0*I, d, d) for i in 1:(m + 1)])

vcm2 = VCModel(Y2, V)

_, obj, niters = vcselect!(vcm; standardize=true)
_, obj1, niters1 = vcselect!(vcm1; standardize=true)

@testset begin 
  @test obj == obj1 
  @test niters == niters1 
  @test vcm.Σ == vcm1.Σ
  @test vcm.β == vcm1.β
end 

resetVCModel!(vcm)
resetVCModel!(vcm1)

@testset begin 
  @test vcm.Σ == [Matrix(1.0*I, d, d) for i in 1:(m+1)]
  @test vcm1.Σ == [Matrix(1.0*I, d, d) for i in 1:(m+1)]
end 

_, obj, niters = vcselect!(vcm; penfun=L1Penalty(), λ=2.5)
_, obj1, niters1 = vcselect!(vcm1; penfun=L1Penalty(), λ=2.5)

@testset begin 
  @test obj == obj1 
  @test niters == niters1 
  @test vcm.Σ == vcm1.Σ
  @test vcm.β == vcm1.β
end 

# path given lambda grid 
resetVCModel!(vcm)
resetVCModel!(vcm1)

Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; 
      penfun=NoPenalty(), λpath=range(1,10,length=20))
      println("Σ̂path=", Σ̂path)
      println("β̂path=", β̂path)
      println("λpath=", λpath)
      println("objpath=", objpath)
      println("niterspath=", niterspath)

Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm1; penfun=L1Penalty(), 
      λpath=range(1,10,length=20))
      println("Σ̂path=", Σ̂path)
      println("β̂path=", β̂path)
      println("λpath=", λpath)
      println("objpath=", objpath)
      println("niterspath=", niterspath)

ranking, = rankvarcomps(Σ̂path)

# path not given lambda grid 
resetVCModel!(vcm)
resetVCModel!(vcm1)

Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; 
      penfun=L1Penalty(), nλ=20)

Σ̂path, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm1; penfun=L1Penalty(), 
      nλ=10)

end # end of module 

