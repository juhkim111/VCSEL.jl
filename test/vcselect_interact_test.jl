module vcselect_interact_test

include("../src/VCSEL.jl")
using Random, LinearAlgebra, StatsBase, .VCSEL, Test

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

# initialize VCintModel 
vcmint = VCintModel(y, V, Vint)
vcmintX = VCintModel(y, X, V, Vint)

# vcselect with no penalty 
@info "check if objective values are monotonically decreasing (no penalty)"
vcmint1 = deepcopy(vcmint)
_, obj, niters, objvec = vcselect!(vcmint1; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (no penalty)"
vcmintX1 = deepcopy(vcmintX)
_, obj, niters, objvec = vcselect!(vcmintX1; verbose=true)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end


# vcselect with L1Penalty
@info "check if objective values are monotonically decreasing (L1 penalty)"
for lambda in range(0, 4.0, length=10)
  vcmint1 = deepcopy(vcmint)
  _, obj, niters, objvec = vcselect!(vcmint1; penfun=L1Penalty(), λ=lambda, verbose=true)
  @testset begin 
      for i in 1:(length(objvec) - 1)
          @test objvec[i] >= objvec[i+1]
      end 
  end 
end

@info "check if objective values are monotonically decreasing (L1 penalty)"
for lambda in range(0, 3.0, length=10)
  vcmintX1 = deepcopy(vcmintX)
  _, obj, niters, objvec = vcselect!(vcmintX1; penfun=L1Penalty(), λ=lambda, verbose=true)
  @testset begin 
      for i in 1:(length(objvec) - 1)
          @test objvec[i] >= objvec[i+1]
      end 
  end 
end

# vcselect with MCPPenalty 
@info "check if objective values are monotonically decreasing (MCP penalty)"
for lambda in range(0, 5.0, length=10)
  vcmint1 = deepcopy(vcmint)
  _, obj, niters, objvec = vcselect!(vcmint1; penfun=MCPPenalty(), λ=lambda, verbose=true)
  @testset begin 
      for i in 1:(length(objvec) - 1)
          @test objvec[i] >= objvec[i+1]
      end 
  end 
end

@info "check if objective values are monotonically decreasing (MCP penalty)"
for lambda in range(0, 3.6, length=10)
  vcmintX1 = deepcopy(vcmintX)
  _, obj, niters, objvec = vcselect!(vcmintX1; penfun=MCPPenalty(), λ=lambda, verbose=true)
  @testset begin 
      for i in 1:(length(objvec) - 1)
          @test objvec[i] >= objvec[i+1]
      end 
  end 
end

# obtain solution path (no penalty)
Σ̂path, Σ̂intpath, β̂path, λpath, objpath, niterspath = vcselectpath!(vcmX; 
      penfun=NoPenalty())
resetModel!(vcmX)
Σ̂path2, Σ̂intpath2, β̂path2, λpath2, objpath2, niterspath2 = vcselectpath!(vcmX)

@testset begin
  @test vcmXΣ == Σ̂path
  @test vcmXΣint == Σ̂intpath
  @test vcmXβ == β̂path
  @test Σ̂path == Σ̂path2
  @test Σ̂intpath == Σ̂intpath2
  @test β̂path == β̂path2
  @test λpath == λpath2
  @test objpath == objpath2
  @test niterspath == niterspath2 
end 

# # reset model 
# resetModel!(vcmint)
# resetModel!(vcmintX)

# # vcselectpath! 
# #Σ̂path, Σ̂intpath, = vcselectpath!(vcmint, penfun=L1Penalty(), nλ=30)
# #Σ̂path, Σ̂intpath, = vcselectpath!(vcmint, penfun=L1Penalty(), λpath=range(0, 0.5, length=10))
# Σ̂path, Σ̂intpath, = vcselectpath!(vcmintX, penfun=L1Penalty(), λpath=range(0, 0.25, length=10))


# println("Σ̂path=$(Σ̂path)")
# println("Σ̂intpath=$(Σ̂intpath)")
# # obtain solution path, not given lambda grid 
# Σ̂path, Σ̂intpath, β̂path, λpath, objpath, niterspath = vcselectpath!(vcm; 
#       penfun=L1Penalty(), nλ=20)
# println("Σ̂path=", Σ̂path)
# println("Σ̂intpath=", Σ̂intpath)
# println("β̂path=", β̂path)
# println("λpath=", λpath)
# println("objpath=", objpath)

# # obtain solution path, given lambda grid
# Σ̂path, Σ̂intpath, β̂path, λpath, objpath, niterspath = vcselectpath!(vcmX; 
#       penfun=L1Penalty(), λpath=range(0, 2, length=5))
# println("Σ̂path=", Σ̂path)
# println("Σ̂intpath=", Σ̂intpath)
# println("β̂path=", β̂path)
# println("λpath=", λpath)
# println("objpath=", objpath)

end # end of module 