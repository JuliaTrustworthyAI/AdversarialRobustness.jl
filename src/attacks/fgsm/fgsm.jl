using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold, logitcrossentropy

# include("../common_utils.jl")

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(
    model,
    x,
    y;
    loss = logitcrossentropy,
    ϵ = 0.3,
    clamp_range = (0, 1),
)
    grads = gradient(
        x -> loss(model(x), y),
        x,
    )
    x = clamp.(x .+ (ϵ .* sign.(grads[1])), clamp_range...)
    return x
end
