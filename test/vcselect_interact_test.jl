module InteractTest

# load packages 
using Random, LinearAlgebra, Test, StatsBase # VarianceComponentSelect
include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

# set seed 
Random.seed!(123)

# generate data from an univariate response variance component model 
n = 100   # no. observations
m = 7    # no. variance components
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
σ2[1] = σ2[5] = 5.0
σ2int[1] = σ2int[5] = 5.0
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
σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, β̂, obj, niters, Ω, objvec = vcselect(y2, X, V, Vint; verbose=true)
println("objvec=", objvec)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; verbose=true, λ=2.0, 
      penfun=L1Penalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, β̂, obj, niters, Ω, objvec = vcselect(y2, X, V, Vint; verbose=true, λ=3.0, 
      penfun=L1Penalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; verbose=true, λ=4.0, 
      penfun=MCPPenalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; verbose=true, λ=1.5, 
      penfun=MCPPenalty())
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

y2new, Vnew, Vintnew, = nullprojection(y2, X, V, Vint)
σ̂2_1, σ̂2int_1, obj, niters, Ω, objvec_1 = vcselect(y2new, Vnew, Vintnew; verbose=true, λ=2.5, 
      penfun=MCPPenalty())

σ̂2_2, σ̂2int_2, β̂, obj, niters, Ω, objvec_2 = vcselect(y2, X, V, Vint; verbose=true, λ=2.5, 
      penfun=MCPPenalty())

@testset begin
  @test σ̂2_1 == σ̂2_2
  @test σ̂2int_1 == σ̂2int_2
  @test objvec_1 == objvec_2
end 
      
@testset begin 
  for i in 1:(length(objvec_1) - 1)
    @test objvec_1[i] >= objvec_1[i+1]
  end 
end 

@testset begin 
  for i in 1:(length(objvec_2) - 1)
    @test objvec_2[i] >= objvec_2[i+1]
  end 
end 

@info "test maxlambda function"
maxλ = maxlambda(y, V, Vint; penfun=L1Penalty())

σ̂2, σ̂2int, obj, niters, Ω, objvec = vcselect(y, V, Vint; verbose=true, λ=maxλ, 
       penfun=L1Penalty())
@testset begin
  @test isapprox(σ̂2[1:end-1], zeros(m); atol=1e-8)
  @test isapprox(σ̂2int, zeros(m); atol=1e-8)
end 

@info "test path function"
σ2path_nopen, σ2intpath_nopen, λpath_nopen, objpath_nopen, niterspath_nopen = 
      vcselectpath(y, V, Vint)

λpath = range(0, 4.0, length=5)
σ2path, σ2intpath, λpath, objpath, niterspath = vcselectpath(y, V, Vint; penfun=L1Penalty(), 
      λpath=λpath)

@testset begin 
  @test σ2path_nopen == σ2path[:, 1]
  @test σ2intpath_nopen == σ2intpath[:, 1]
  @test objpath_nopen == objpath[1]
  @test niterspath_nopen == niterspath[1]
  @test isempty(λpath_nopen)
end 

λpath = range(0, 4.0, length=5)
σ2path, σ2intpath, βpath, objpath, λpath, niterspath = vcselectpath(y2, X, V, Vint; 
      penfun=L1Penalty(), λpath=λpath)

σ2path, σ2intpath, βpath, objpath, λpath, niterspath = vcselectpath(y2, X, V, Vint; 
      penfun=L1Penalty(), nλ=10)

end 