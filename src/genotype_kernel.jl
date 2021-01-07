"""
    genotype_kernel(Gobs, weights_beta, geno_kernel)

Constructs genotype kernel based on options: G[i] = Gobs[i] * W * I if geno_kernel = "SKAT"
OR G[i] = Gobs[i] * W * ones(qi) if geno_kernel = "Burden" where W is a diagonal matrix whose
entries are weights. Each G[i] is divided by square root of frobenius norm of G[i]*transpose(G[i]).

# Input
- obs: 
- weights_beta: 
- geno_kernel: "Burden" or "SKAT"

# Output 
- `G`: 
"""
function genotype_kernel(
    Gobs  :: AbstractVector{Matrix{T}},
    weights_beta :: AbstractVector{S},
    geno_kernel :: AbstractString 
    ) where {T, S <: Real}

    # handle errors 
    @assert length(weights_beta) == 2 "weights_beta takes only two parameters for beta distribution!\n"
    @assert geno_kernel ∈ ["Burden", "SKAT"] "geno_kernel must be either Burden or SKAT!\n"

    # 
    G = similar(Gobs)
    if geno_kernel == "SKAT"
        for i in eachindex(Gobs)
            maf = mean(Gobs[i], dims=1)[:] ./ 2
            G[i] = Gobs[i][:, maf .> 0]
            # weight vector 
            W = diagm(pdf.(Beta(weights_beta[1], weights_beta[2]), maf[maf .> 0]))
            G[i] *= W
            BLAS.syrk!('U', 'T', true, G[i], false, W)
            G[i] ./= sqrt(norm(Symmetric(W)))
        end 
    elseif geno_kernel == "Burden"
        for i in eachindex(Gobs)
            # calculate MAF by column mean 
            maf = mean(Gobs[i], dims=1)[:] ./ 2
            # exclude columns whose MAF equals to 0 
            G[i] = Gobs[i][:, maf .> 0]
            # weight vector 
            w = pdf.(Beta(weights_beta[1], weights_beta[2]), maf[maf .> 0])
            # 
            G[i] *= reshape(w, :, 1)
            G[i] ./= sqrt(dot(G[i], G[i]))
            # divide by square root of frobenius norm of G[i] * G[i]'
        end 
    end

    return G 

end
