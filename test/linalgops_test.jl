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

# cast to actual real matrix: this implementation very wasteful of memory
function Base.copyto!(M::Matrix{T}, Ω::MvcCovMatrix{T}) where T <: Real
    n, d, m = size(Ω.G[1], 1), size(Ω.Σ[1], 1), length(Ω.G)
    M .= kron(Ω.Σ[end], I(n))
    for i in 1:m
        M .+= kron(Ω.Σ[i], Ω.G[i] * transpose(Ω.G[i]))
    end
    M
end

Sigma = [generate_psd(d) for i in 1:m+1]
G = [rand([0.0, 1.0, 2.0], n, q[i]) for i in 1:m]
Ω = MvcCovMatrix(Sigma, G)
#Ωmat = Matrix(Ω)

println("memory usage of Ω (GB): $(Base.summarysize(Ω) / 1e9)")
#println("memory usage of Ωmat (GB): $(Base.summarysize(Ωmat) / 1e9)")


# 
Y = randn(n, d)
p = 3
X = randn(n, p)
vecR = cg(Ω, Y)

# Ωchol = cholesky!(Symmetric(Ωmat))
# Ωinv = inv(Ωchol) 
# ΩinvY = Ωinv * vec(Y)  

M = randn(n * d, 5)
@info "check correctness of mul!"
#@show norm(vecR - ΩinvY)
#@show norm(Ω * M - Ωmat * M)


function formM(G::Matrix{T}, Ωmat::MvcCovMatrix{T}, d::Int) where T <: Real
    Ωmat = Matrix(Ω)
    Ωchol = cholesky!(Symmetric(Ωmat))
    Ωinv = inv(Ωchol) 
    n = size(G, 1)
    kron_I_one = kron(Matrix(I, d, d), ones(n)) # dn x d
    Mndxnd = kron(ones(d, d), G * G') # 
    Mndxnd .*= Ωinv
    Mdxd = BLAS.gemm('T', 'N', kron_I_one, Mndxnd * kron_I_one)
    return Mdxd
end 

Mdxd_2 = Matrix{Float64}(undef, d, d)
#rhs_nd_q = Matrix{Float64}(undef, n * d, )
# storage_nd_q = Matrix{Float64}(undef, n * d, maximum([size(G[i], 2) for i in 1:m]))

@info "check correctness of updateM!"
#for i in 1:m 
i=1
@time M1 = formM(G[i], Ω, d)
@time M2 = updateM!(Mdxd_2, G[i], Ω)
@show norm(M1 - M2)
#end 

# @info "check nullprojection"
# ynew, Gnew, B = nullprojection(Y, X, G)
# @testset begin 
#     @test isapprox(norm(X' * B), 0; atol=10)
#     @test size(ynew, 1) == size(Gnew[1], 1)
# end