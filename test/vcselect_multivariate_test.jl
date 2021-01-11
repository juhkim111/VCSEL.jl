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

@info "initialize VCModel for multivariate trait"
vcm = VCModel(Y, X, G)

# i=6
# function updateΣ(
#   vcm    :: VCModel, 
#   Gi     :: AbstractMatrix{T}, 
#   i      :: Int;
#   penfun :: Penalty = NoPenalty(),
#   λ      :: Real = zero(T)
# ) where {T <: Union{Real, Bool}}
  
#   n, d = size(vcm)[1], size(vcm)[2]
#   tmp = kron(ones(d, d), Gi * Gi')
#   tmp .*= vcm.Ωinv
#   kron_I_ones = kron(I(d), ones(n))
#   tmp2 = kron_I_ones' * tmp * (kron_I_ones)
#   tmp2chol = cholesky!(Symmetric(tmp2))
#   tmp3 = Gi' * vcm.R * vcm.Σ[i] * tmp2chol.L 
#   tmp4 = tmp3' * tmp3 
#   tmp5 = sqrt(tmp4)
#   Linv = inv(tmp2chol.L)
#   Σ = Linv' * tmp5 * Linv 
#   return Σ
# end 
# @time tmp = updateΣ(vcm, I(size(vcm)[1]), i)
# println("tmp=", tmp)


vcm = VCModel(Y, X, G)

_, obj, niters, objvec = vcselect!(vcm; penfun=NoPenalty())

@info "objective values are monotonically decreasing (no penalty)" 
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

vcm2 = VCModel(Y, X, G)
_, _, _, objvec = vcselect!(vcm2; penfun=L1Penalty(), λ=5.0)

@info "objective values are monotonically decreasing (L1 penalty)" 
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 

vcm3 = VCModel(Y, G)
_, _, _, objvec = vcselect!(vcm3; penfun=MCPPenalty(), λ=6.0)

@info "objective values are monotonically decreasing (MCP penalty)" 
@testset begin 
  for i in 1:(length(objvec) - 1)
    @test objvec[i] >= objvec[i+1]
  end 
end 


# weights for adaptive lasso penalty 
penwt = 1 ./ sqrt.(tr.(vcm.Σ))
penwt[end] = 0.0

@info "objective values are monotonically decreasing (adaptive L1 penalty)" 
for lambda in range(70,0, length=10)
  vcm_tmp = VCModel(Y, X, G)
  _, _, _, objv = vcselect!(vcm_tmp; penfun=L1Penalty(), λ=lambda, penwt=penwt)
  @testset begin 
    for i in 1:(length(objv) - 1)
        @test objv[i] >= objv[i+1]
    end 
  end 
end


resetModel!(vcm)
@info "vcselect path"
Σ̂path, λpath, objpath, niterspath = vcselectpath!(vcm; penfun=L1Penalty(), nλ=30)
@testset begin 
  @test Σ̂path[:, 1] != Σ̂path[:, end]
  @test length(λpath) == 30
  @test length(objpath) == 30
  @test length(niterspath) == 30
end 
