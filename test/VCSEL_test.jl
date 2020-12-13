# Generate a sample covariance matrix
Random.seed!(123)
n = 500
d = 3
m = 20
q = rand(5:50, m) # #snps in each gene

# generate random positive semidefinite matrix 
function generate_psd(d::Integer)
    A = rand(d, d)
    A'A
end

Σ = [generate_psd(d) for i in 1:m+1]
G = [rand([0.0, 1.0, 2.0], n, q[i]) for i in 1:m]
Ω = MvcCovMatrix(Σ, G)

# 
Y = randn(n, d)
p = 3
X = randn(n, p)

vcm = VCModel(Y, X, G)
@testset begin
    @test length(vcm) == d
    @test ncovariates(vcm) == p
    @test size(vcm) == (n, d)
    @test nmeanparams(vcm) == p*d
    @test nvarcomps(vcm) == (m + 1)
    @test ngroups(vcm) == m 
end 

vcm2 = VCModel(Y, G)
@testset begin
    @test length(vcm2) == d
    @test ncovariates(vcm2) == 0
    @test size(vcm2) == (n, d)
    @test nmeanparams(vcm2) == 0
    @test nvarcomps(vcm2) == (m + 1)
    @test ngroups(vcm2) == m 
end 