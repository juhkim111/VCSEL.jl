# Generate a sample covariance matrix
Random.seed!(123)
n = 300
d = 3
m = 10
p = 3
q = rand(5:50, m) # #snps in each gene

# generate random positive semidefinite matrix 
function generate_psd(d::Integer)
    A = rand(d, d)
    A'A
end

Σ = [generate_psd(d) for i in 1:m+1]
G = [rand([0.0, 1.0, 2.0], n, q[i]) for i in 1:m]
Y = randn(n, d)
X = randn(n, p)

Gker_skat = genotype_kernel(G, [1, 25], "SKAT")
Gker_burden = genotype_kernel(G, [1, 25], "Burden")

@testset "test genotype kernel function" for i in 1:m
    maf = mean(G[i], dims=1)[:] ./ 2
    Gtmp = G[i][:, maf .> 0]
    # weight vector 
    w = pdf.(Beta(1,25), maf[maf .> 0]) 
    # test gentoype kernel: SKAT 
    Vi = Gtmp * (Diagonal(w)).^2 * Gtmp' 
    Vi ./= norm(Vi)
    @test Gker_skat[i] * Gker_skat[i]' ≈ Vi 
    @test isapprox(norm(Gker_skat[i] * Gker_skat[i]'), 1, atol=1e-7)
    # test gentoype kernel: Burden
    Vi = Gtmp * Diagonal(w) * ones(length(w), length(w)) * Diagonal(w) * Gtmp'
    Vi ./= norm(Vi)
    @test Gker_burden[i] * Gker_burden[i]' ≈ Vi 
    @test isapprox(norm(Gker_burden[i] * Gker_burden[i]'), 1, atol=1e-7)
end