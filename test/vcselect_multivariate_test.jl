## generate data from a 3-variate response variance component model
Random.seed!(123)
n = 100         # no. observations
d = 3           # no. categories
m = 5           # no. variance components
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
  V[i] ./ norm(V[i])
end
V[end] = Matrix(1.0*I, n, n) / sqrt(n)

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

_, obj, niters, objvec = vcselect!(vcm)
_, obj1, niters1, objvec1 = vcselect!(vcm1)

@info "initialize VCModel for multivariate trait"
@testset begin 
  @test obj == obj1 
  @test niters == niters1 
  @test vcm.Σ == vcm1.Σ
  @test vcm.β == vcm1.β
end 

@info "objective values are monotonically decreasing (no penalty)" 
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

resetModel!(vcm)
resetModel!(vcm1)

@testset begin 
  @test vcm.Σ == [Matrix(1.0*I, d, d) for i in 1:(m+1)]
  @test vcm1.Σ == [Matrix(1.0*I, d, d) for i in 1:(m+1)]
end 

_, obj, niters, objvec = vcselect!(vcm; penfun=L1Penalty(), λ=2.5)
_, obj1, niters1, objvec1 = vcselect!(vcm1; penfun=L1Penalty(), λ=2.5)

@testset begin 
  @test obj == obj1 
  @test niters == niters1 
  @test vcm.Σ == vcm1.Σ
  @test vcm.β == vcm1.β
end 

@info "objective values are monotonically decreasing (L1 penalty)" 
for lambda in range(70,0,length=10)
  vcm = VCModel(Y, X, V)
  _, obj, niters, objvec = vcselect!(vcm; penfun=L1Penalty(), λ=lambda)
  @testset begin 
      for i in 1:(length(objvec) - 1)
          @test objvec[i] >= objvec[i+1]
      end 
  end 
end


@info "objective values are monotonically decreasing (MCP penalty)" 
for lambda in range(70,0,length=10)
  vcm = VCModel(Y, X, V)
  _, obj, niters, objvec = vcselect!(vcm; penfun=MCPPenalty(), λ=lambda)
  @testset begin 
    for i in 1:(length(objvec) - 1)
        @test objvec[i] >= objvec[i+1]
    end 
  end 
end

# weights for adaptive lasso penalty 
vcm0 = deepcopy(vcm)
vcselect!(vcm0; penfun=NoPenalty())
penwt = 1 ./ sqrt.(tr.(vcm0.Σ))
penwt[end] = 0.0

@info "objective values are monotonically decreasing (adaptive L1 penalty)" 
for lambda in range(70,0,length=10)
  vcm = VCModel(Y, X, V)
  _, obj, niters, objvec = vcselect!(vcm; penfun=L1Penalty(), λ=lambda, penwt=penwt)
  @testset begin 
    for i in 1:(length(objvec) - 1)
        @test objvec[i] >= objvec[i+1]
    end 
  end 
end

