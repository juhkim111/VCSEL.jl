# Generate a sample covariance matrix
Random.seed!(123)
n = 100
d = 2
m = 5
p = 3
X = randn(n, p)    # covariate matrix 
β = ones(p, d)     # fixed effects parameter matrix 
q = rand(5:20, m)  # #snps in each gene

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


# find maximum lambda and test 
@info "test findmaxλ with L1Penalty"
vcm = VCModel(Y, X, G)
Σinit = deepcopy(vcm.Σ)
maxλ_lasso, iter = findmaxλ(vcm; penfun=L1Penalty())
@show maxλ_lasso
vcselect!(vcm; penfun=L1Penalty(), λ=maxλ_lasso)
@testset begin 
    for i in 1:m
      @test isapprox(vcm.Σ[i], zeros(d, d); atol=1e-6)
    end 
end 

@info "test findmaxλ with MCPPenalty"
resetModel!(vcm, Σinit)
maxλ_mcp, iter = findmaxλ(vcm; penfun=MCPPenalty())
vcselect!(vcm; penfun=MCPPenalty(), λ=maxλ_mcp)
@show maxλ_mcp
@testset begin 
  for i in 1:m
    @test isapprox(vcm.Σ[i], zeros(d, d); atol=1e-6)
  end 
end 

@info "test findmaxλ with NoPenalty"
resetModel!(vcm, Σinit)
maxλ, iter = findmaxλ(vcm; penfun=NoPenalty())
@show maxλ
@testset begin 
  @test maxλ == 0
  @test iter == 0
end 

