using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

# include("../common_utils.jl")

# White-box Projected Gradient Descent (PGD) attack by Madry et al. (arxiv.org/abs/1706.06083)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function PGD(
    model,
    x,
    y;
    loss = cross_entropy_loss,
    ϵ = 0.3,
    step_size = 0.01,
    iterations = 40,
    min_label = 0,
    max_label = 9,
    clamp_range = (0, 1),
)
    w, h, c = size(x)
    x_curr = deepcopy(x)
    x_curr = reshape(x_curr, w, h, c, 1)
    x_curr_adv =
        clamp.(
            x_curr + (randn(Float32, size(x_curr)...) * Float32(step_size)),
            clamp_range...,
        )
    iteration = 1
    δ = chebyshev(x_curr, x_curr_adv)

    while (δ .< ϵ) && iteration <= iterations
        x_curr_adv = FGSM(
            model,
            x_curr_adv,
            y;
            loss = loss,
            ϵ = step_size,
            min_label = min_label,
            max_label = max_label,
            clamp_range = clamp_range,
        )
        iteration += 1
        δ = chebyshev(x_curr, x_curr_adv)
    end

    return clamp.(x_curr_adv, x_curr .- ϵ, x_curr .+ ϵ)

    # for i = 1:batch_size
    #     x_curr = x_adv[:, :, :, i]
    #     x_curr = reshape(x_curr, 28, 28, 1, 1)
    #     x_curr_adv = clamp.(x_curr + (randn(Float32, size(x_curr)...) * Float32(step_size)), clamp_range...); # start from the random point
    #     δ = chebyshev(x_curr, x_curr_adv)
    #     iteration = 1; while (δ < ϵ) && iteration <= iterations
    #         x_curr_adv = FGSM(model, x_curr_adv, y; loss=loss, ϵ = step_size, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
    #         δ = chebyshev(x_curr, x_curr_adv)
    #         iteration += 1
    #     end
    #     x_adv[:, :, :, i] = x_curr_adv
    # end
    # return x_adv
end
