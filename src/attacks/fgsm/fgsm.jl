using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold, logitcrossentropy

# include("../common_utils.jl")

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(model, x, y; loss=logitcrossentropy, ϵ = 0.3, min_label=0, max_label=9, clamp_range = (0, 1))
    w, h, c, batch_size = size(x)
    # x_adv = deepcopy(x)
    x_adv = reshape(x, w, h, c, batch_size)

    # grads = zeros(size(x)) # size(x) = 28, 28, 1, batch_size

    # With a batch: scalar indexing?
    # grads = Flux.jacobian((x_adv) -> loss(model(x_adv), y; agg=(x->x)), x_adv)[1]
    # grads = reshape(grads, 11, 28, 28, 1, 11)
    # grads_real = zeros(size(x))
    # i = 1
    # for i = 1:batch_size
    #     grads_real[:, :, :, i] = grads[i, :, :, :, i]
    # end
    # x_adv = clamp.(x_adv .+ (ϵ .* sign.(grads_real)), clamp_range...)

    # To test with batch size of one
    # for i = 1:batch_size
    #     curr_input = reshape(x_adv[:, :, :, i], w, h, c, 1)
    #     curr_label = y[:, i]
    #     grads[:, :, :, i] = Flux.gradient((curr_input) -> loss(model(curr_input), curr_label; agg=(x->x))[1], curr_input)[1]
    # end
    grads = gradient((x_adv) -> loss(model(x_adv), y), x_adv)
    x_adv = clamp.(x_adv .+ (ϵ .* sign.(grads[1])), clamp_range...)

    return x_adv
end