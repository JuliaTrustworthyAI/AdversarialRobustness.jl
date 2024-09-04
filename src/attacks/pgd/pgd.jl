using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold, logitcrossentropy

# include("../common_utils.jl")

# White-box Projected Gradient Descent (PGD) attack by Madry et al. (arxiv.org/abs/1706.06083)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function PGD(model, x, y; loss=logitcrossentropy, ϵ = 0.3, step_size = 0.01, iterations = 40, min_label=0, max_label=9, clamp_range = (0, 1))
    w, h, c, batch_size = size(x)
    # x_curr = deepcopy(x)
    x_curr = reshape(x, w, h, c, batch_size)
    x_curr_adv = clamp.(x_curr + ((randn(Float32, size(x_curr)...) |> gpu) * Float32(step_size)), clamp_range...)
    iteration = 1
    # δ = chebyshev(x_curr, x_curr_adv)

    # (δ .< ϵ) && 

    while iteration <= iterations
        x_curr_adv = FGSM(model, x_curr_adv, y; loss=loss, ϵ = step_size, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
        iteration += 1
        # δ = chebyshev(x_curr, x_curr_adv)
    end

    return clamp.(x_curr_adv, x_curr .- ϵ, x_curr .+ ϵ)
end