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

@testset "test projection onto null space of X" begin
  Ynew, Gnew, B = nullprojection(Y, X, G)
  @test B'B ≈ I 
  @test isapprox(maximum(abs.(B'*X)), 0; atol=1e-8) #all(B'*X .≈ 0)
  for i in 1:m 
    @test size(Gnew[i], 1) * d == size(Ynew, 1)
  end
end 

Ω = Matrix{Float64}(undef, n*d, n*d);
#@testset "test forming Ω" begin
formΩ!(Ω, Σ, G)

Ω2 = Matrix{Float64}(undef, n*d, n*d);

function formΩ2!(
    Ω :: AbstractMatrix{T}, 
    Σ :: AbstractVector{Matrix{T}}, 
    G :: AbstractVector{Matrix{T}}
    ) where {T <: Real}
    n = size(G[1], 1)
    Ω .= kron(Σ[end], I(n))
    Ω ./= √n
    for i in 1:m
        kronaxpy!(Σ[i], G[i] * transpose(G[i]), Ω)
    end
end 
@btime formΩ2!(Ω2, Σ, G)

@info "test estimates of Ω"
@test isapprox(norm(Symmetric(Ω) - Ω2), 0; atol=1e-8)


