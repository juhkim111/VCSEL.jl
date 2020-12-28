## generate data from a 3-variate response variance component model
Random.seed!(123)
n = 100         # no. observations
d = 3           # no. categories
m = 5           # no. variance components
p = 4           # no. covariates
X = randn(n, p) # covariate matrix 
β = ones(p, d)  # fixed effects parameter matrix 
q = rand(5:50, m) # #snps in each gene

# generate random positive semidefinite matrix 
function generate_psd(d::Integer)
  A = rand(d, d)
  A'A
end

# generate variance component matrix 
Σ = [generate_psd(d) for i in 1:m+1]
# generate genotype matrices 
G = [rand([0.0, 1.0, 2.0], n, q[i]) for i in 1:m]

# form Ω
Ω = Matrix{Float64}(undef, n*d, n*d)
formΩ!(Ω, Σ, G)
Ωchol = cholesky!(Symmetric(Ω))

# generate response vector (no covariate matrix)
Y = X * β + reshape(Ωchol.L * randn(n*d), n, d)
Y2 = reshape(Ωchol.L * randn(n*d), n, d)

# construct VCModel 
vcm = VCModel(Y, X, G)

# perform variance component selection 
vcselect!(vcm; penfun=L1Penalty(), λ=3.0, maxiters=100, tol=1e-5)
@show vcm.Σ

# tests 
@info "check basic functions"
@testset begin
  @test length(vcm) == d
  @test ncovariates(vcm) == p
  @test nvarcomps(vcm) == (m + 1)
end

@info "check resetModel! function"
Ωtmp = similar(vcm.Ωinv)
resetModel!(vcm)
formΩ!(Ωtmp, vcm.Σ, vcm.G)
cholΩtmp = cholesky!(Symmetric(Ωtmp))
@testset begin 
  @test vcm.Σ == [I(d) for i in 1:m+1]
  @test vcm.Ωinv == inv(cholΩtmp)
end 
resetModel!(vcm, Σ)
formΩ!(Ωtmp, vcm.Σ, vcm.G)
cholΩtmp = cholesky!(Symmetric(Ωtmp))
@testset begin 
  @test vcm.Σ == Σ
  @test vcm.Ωinv == inv(cholΩtmp)
end 
resetModel!(vcm)
vcselect!(vcm; penfun=L1Penalty(), λ=3.0, maxiters=100, tol=1e-5)
@show vcm.Σ
