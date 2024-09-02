using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

# include("../common_utils.jl")

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(
    model,
    x,
    y;
    loss = cross_entropy_loss,
    ϵ = 0.3,
    min_label = 0,
    max_label = 9,
    clamp_range = (0, 1),
)
    w, h, c = size(x)
    x_adv = deepcopy(x)
    x_adv = reshape(x_adv, w, h, c, 1)
    grads = gradient(
        x_adv -> loss(model(x_adv), y; min_label = min_label, max_label = max_label)[1],
        x_adv,
    )[1]
    x_adv = clamp.(x_adv .+ (ϵ .* sign.(grads)), clamp_range...)
    return x_adv
end
