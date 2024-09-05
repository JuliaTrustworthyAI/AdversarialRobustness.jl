using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold, logitcrossentropy

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
    clamp_range = (0, 1),
)

    xadv =
        clamp.(
            x + (randn(Float32, size(x)...) * Float32(step_size)),
            clamp_range...,
        )
    iteration = 1
    δ = chebyshev(x, xadv)

    while (δ .< ϵ) && iteration <= iterations
        xadv = FGSM(
            model,
            xadv,
            y;
            loss = loss,
            ϵ = step_size,
            clamp_range = clamp_range,
        )
        iteration += 1
        δ = chebyshev(x, xadv)
    end

    return clamp.(xadv, x .- ϵ, x .+ ϵ)
end
