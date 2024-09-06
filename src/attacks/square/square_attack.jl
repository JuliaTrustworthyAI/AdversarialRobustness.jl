using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

include("utils.jl")

# The black-box Square Attack developed by Andriuschenko et al. (https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)
# The only free variable is the budget: iterations
function SquareAttack(
    model,
    x,
    y;
    iterations = 10,
    ϵ = 0.3,
    p_init = 0.8,
    verbose = false,
    clamp_range = (0, 1)
)
    Random.seed!(0)
    n_features = length(x)
    w, h, c, _ = size(x)

    # Initialization (stripes of +/-ϵ)
    init_δ = rand(w, 1, c) .|> x -> x < 0.5 ? -ϵ : ϵ
    init_δ_extended = repeat(init_δ, 1, h, 1)
    x_best = clamp.((init_δ_extended + x), clamp_range...)

    topass_x_best = deepcopy(x_best)

    logits = model(topass_x_best)
    loss_min = margin_loss(logits, y)
    margin_min = margin_loss(logits, y)
    n_queries = 1

    for iteration = 1:iterations
        fooled = margin_min[1] < 0

        if iteration == 1 && verbose
            println("preds: ", (onecold(logits) .- 1))
            println("margin min: ", margin_min)
            println("fooled? ", fooled)
            println()
        end

        # Data point misclassified
        if fooled
            break
        end

        x_curr, x_best_curr, y_curr = x, x_best, y
        loss_min_curr, margin_min_curr = loss_min, margin_min
        δ = x_best_curr .- x_curr

        p = p_selection(p_init, iteration, iterations)
        s = Int(round(sqrt(p * n_features / c)))
        s = min(max(s, 1), h)

        center_h = rand(1:(h-s))
        center_w = rand(1:(w-s))

        # values = rand([-2ϵ, 2ϵ], c)
        values = rand([-ϵ, ϵ], c)

        δ[center_w:center_w+s-1, center_h:center_h+s-1, :] .= values

        x_new = clamp.(x_curr .+ δ, clamp_range...)

        topass_x_new = deepcopy(x_new)
        # topass_x_new = reshape(topass_x_new, size(x)..., 1)

        logits = model(topass_x_new)
        loss = margin_loss(logits, y_curr)
        margin = margin_loss(logits, y_curr)

        if loss[1] < loss_min_curr[1]
            loss_min = loss
            margin_min = margin
            x_best = x_new
        end

        n_queries += 1
    end

    return x_best, n_queries
end
