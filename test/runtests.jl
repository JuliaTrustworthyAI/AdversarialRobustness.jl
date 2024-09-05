using AdversarialRobustness
using Test

@testset "AdversarialRobustness.jl" begin
    include("aqua.jl")
    include("train_mlp.jl")
end
