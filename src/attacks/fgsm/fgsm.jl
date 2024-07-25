using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

# include("../common_utils.jl")

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(model, x, y; loss=cross_entropy_loss, ϵ = 0.3, min_label=0, max_label=9, clamp_range = (0, 1))
    batch_size = size(x)[4]
    x_adv = deepcopy(x)
    for i = 1:batch_size
        x_curr = x_adv[:, :, :, i]
        x_curr = reshape(x_curr, 28, 28, 1, 1)
        grads = gradient(x_curr -> loss(model(x_curr), y[i]; min_label=min_label, max_label=max_label)[1], x_curr)[1]
        x_curr_adv = clamp.(x_curr .+ (ϵ[i] .* sign.(grads)), clamp_range...)
        x_adv[:, :, :, i] = x_curr_adv
    end
    return x_adv
end