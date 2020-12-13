using BenchmarkTools, LinearAlgebra, Random, Revise
import Base: copyto!

struct MvcCovMatrix{T <: Real} <: AbstractMatrix{T}
    Σ  :: Vector{Matrix{T}} # m + 1 variance components: Σ1, ..., Σm, Σ0
    G  :: Vector{Matrix{T}} # m genotype matrices: G1, ..., Gm
    # working arrays
    storage_nd   :: Matrix{T}
    storage_qd_1 :: Matrix{T}
    storage_qd_2 :: Matrix{T}
end

# constructor
function MvcCovMatrix(Σ::Vector{Matrix{T}}, G::Vector{Matrix{T}}) where T <: Real
    n, d, m      = size(G[1], 1), size(Σ[1], 1), length(G)
    storage_nd   = Matrix{T}(undef, n, d)
    storage_qd_1 = Matrix{T}(undef, maximum([size(G[i], 2) for i in 1:m]), d)
    storage_qd_2 = similar(storage_qd_1)
    MvcCovMatrix{T}(Σ, G, storage_nd, storage_qd_1, storage_qd_2)
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

LinearAlgebra.issymmetric(::MvcCovMatrix) = true
Base.size(M::MvcCovMatrix) = 
    (size(Σ[1], 1) * size(G[1], 1), size(Σ[1], 1) * size(G[1], 1))

# Generate a sample covariance matrix
Random.seed!(123)
n = 5_000
d = 3
m = 20
q = rand(5:50, m) # #snps in each gene

function generate_psd(d::Integer)
    A = rand(d, d)
    A'A
end

Σ = [generate_psd(d) for i in 1:m+1]
G = [rand([0.0, 1.0, 2.0], n, q[i]) for i in 1:m]
Ω = MvcCovMatrix(Σ, G)
Ωmat = Matrix(Ω)

println("memory usage of Ω (GB): $(Base.summarysize(Ω) / 1e9)")
println("memory usage of Ωmat (GB): $(Base.summarysize(Ωmat) / 1e9)")

function LinearAlgebra.mul!(
    res :: AbstractVecOrMat{T}, 
    Ω   :: MvcCovMatrix{T}, 
    M   :: AbstractVecOrMat{T}) where T <: Real
    n, d, m = size(Ω.G[1], 1), size(Ω.Σ[1], 1), length(Ω.G)
    Σ, G = Ω.Σ, Ω.G
    for col in 1:size(M, 2)
        Mcol = reshape(view(M, :, col), n, d)
        rcol = view(res, :, col)
        mul!(Ω.storage_nd, Mcol, Σ[end])
        copyto!(rcol, Ω.storage_nd)
        for i in 1:m
            q = size(G[i], 2)
            mul!(view(Ω.storage_qd_1, 1:q, :), transpose(G[i]), Mcol)
            mul!(view(Ω.storage_qd_2, 1:q, :), view(Ω.storage_qd_1, 1:q, :), Σ[i])
            mul!(Ω.storage_nd, G[i], view(Ω.storage_qd_2, 1:q, :))
            @inbounds @simd for j in length(rcol)
                rcol += Ω.storage_nd[j]
            end 
        end
    end
    res
end

M = randn(n * d, 5)
@info "check correctness"
@show norm(Ω * M - Ωmat * M)

@info "Benchmark Ω * M"
out = similar(M)
bm = @benchmark mul!($out, $Ω, $M)
display(bm)

@info "Benchmark Ωmat * M"
bm = @benchmark mul!($out, $Ωmat, $M)
display(bm)


function 