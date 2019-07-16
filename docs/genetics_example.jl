using JLD2
@load "/Users/juhyun-kim/Box Sync/workspace/vcselect/codebase/julia/VarianceComponentSelect.jl/docs/tutorial1.jld2" m y X V

include("../src/VarianceComponentSelect.jl")
using .VarianceComponentSelect

σ2 = ones(m + 1)
vcm = VCModel(y, X, V, σ2)

σ̂2, β̂, obj, niters, = vcselect!(vcm)
println(σ̂2)

typeof(vcm)
resetVCModel!(vcm)
println(vcm.Σ)
resetVCModel!(vcm, σ̂2)
println(vcm.Σ)