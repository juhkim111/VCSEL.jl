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

@info "objective values are monotonically decreasing (no penalty)" 
vcm11 = deepcopy(vcm1)
_, _, _, objvec = vcselect!(vcm11)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 
temp1 = vcm11.Σ

vcm21 = deepcopy(vcm2)
_, _, _, objvec = vcselect!(vcm21)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 
temp2 = vcm21.Σ

@info "objective values are monotonically decreasing (L1 Penalty)"
vcm22 = deepcopy(vcm2)
_, _, _, objvec = vcselect!(vcm22; 
      penfun=L1Penalty(), λ=2.0)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (adaptive L1 Penalty)" 
vcm12 = deepcopy(vcm1)
penwt = zeros(m + 1)
penwt[1:m] = 1 ./ sqrt.(temp1[1:m])
_, _, _, objvec = vcselect!(vcm12; penwt=penwt, λ=5.0)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (adaptive L1 penalty)" 
vcm22 = deepcopy(vcm2)
penwt = zeros(m + 1)
penwt[1:m] = 1 ./ sqrt.(temp2[1:m])
_, _, _, objvec = vcselect(y, V; penfun=L1Penalty(), λ=15.0, penwt = penwt)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

@info "check if objective values are monotonically decreasing (MCP penalty)" 
vcm12 = deepcopy(vcm1)
_, _, _, objvec = vcselect!(vcm12; penfun=MCPPenalty(), λ=5.0)
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 


@info "check if vcselect! and vcselectpath! are equivalent"
resetModel!(vcm2)
nλ = 10
Σ̂path2, β̂path2, λpath, = vcselectpath!(vcm2; penfun=L1Penalty(), nλ = nλ)

resetModel!(vcm2)
Σ̂path = Array{Float64}(undef, m+1, nλ)
β̂path = zeros(Float64, p, nλ)
#λpath = range(0, 7.0, length=nλ)
for iter in 1:nλ
    vcselect!(vcm2; penfun=L1Penalty(), λ=λpath[iter], checktype=false)
    Σ̂path[:, iter] .= vcm2.Σ
    β̂path[:, iter] .= vcm2.β
end

@testset begin 
  @test Σ̂path == Σ̂path2
  @test β̂path == β̂path2
end 