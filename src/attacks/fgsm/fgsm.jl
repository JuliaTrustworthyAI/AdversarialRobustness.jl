using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

# include("../common_utils.jl")

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(model, x, y; loss=cross_entropy_loss, ϵ=0.3, clamp_range=(0, 1))
    grads = gradient(x -> loss(model(x), y), x)[1]
    x = clamp.(x .+ (ϵ .* sign.(grads)), clamp_range...)
    return x
end
