using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

include("utils.jl")
# include("../common_utils.jl")

# The black-box Square Attack developed by Andriuschenko et al. (https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)
# The only free variable is the budget: iterations
function SquareAttack(model, x, y, iterations; ϵ=0.3, p_init=0.8, min_label=0, max_label=9, verbose=false, clamp_range = (0, 1))
    Random.seed!(0)
    w, h, c, n_ex_total = size(x)
    n_features = c*h*w

    # Initialization (stripes of +/-ϵ)
    init_δ = rand(w, 1, c, n_ex_total) .|> x -> x < 0.5 ? -ϵ : ϵ
    init_δ_extended = repeat(init_δ, 1, h, 1, 1)
    x_best = clamp.((init_δ_extended + x), clamp_range...)

    logits = model(x_best)
    loss_min = margin_loss(logits, y, min_label, max_label)
    margin_min = margin_loss(logits, y, min_label, max_label)
    n_queries = ones(Float64, n_ex_total)

    times_it_actually_improved = 0

    for iteration = 1:iterations
        idx_to_fool = findall(x -> margin_min[x] > 0, 1:n_ex_total) # attacking images that are predicted correctly

        if iteration == 1 && verbose
            println("n_ex_total: ", n_ex_total)
            println("preds: ", (onecold(logits) .- 1))
            println("margin mins: ", margin_min)
            println("indexes chosen to fool: ", idx_to_fool)
            println()
        end

        # Nothing to fool - all datapoints misclassified
        if length(idx_to_fool) == 0
            break
        end

        x_curr, x_best_curr, y_curr = x, x_best, y
        loss_min_curr, margin_min_curr = loss_min, margin_min
        δ = x_best_curr .- x_curr

        p = p_selection(p_init, iteration, iterations)
        s = Int(round(sqrt(p * n_features/c)))
        s = min(max(s, 1), h)

        for i_img in idx_to_fool
            center_h = rand(1:(h - s))
            center_w = rand(1:(w - s))

            # values = rand([-2ϵ, 2ϵ], c)
            values = rand([-ϵ, ϵ], c)

            # -1 because 
            δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img] .= values 

            # In the SquareAttack paper's actual code but not in advertorch so commenting out for now

            # x_curr_window = x_curr[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
            # x_best_curr_window = x_best_curr[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
            # while check_delta(δ, x_curr_window, x_best_curr_window, center_w, center_h, s, c, i_img, min_val, max_val)
            #     while_entries += 1
            #     random_choices = rand([0, 1], c)
            #     values = (random_choices .* 2 .- 1) .* ϵ 
            #     δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img] .= values 
            # end
        end

        x_new = clamp.(x_curr + δ, clamp_range...)

        logits = model(x_new)
        loss = margin_loss(logits, y_curr, min_label, max_label)
        margin = margin_loss(logits, y_curr, min_label, max_label)

        idx_improved = findall(x -> loss[x] < loss_min_curr[x], 1:n_ex_total)
        times_it_actually_improved += length(idx_improved)

        for i in idx_to_fool
            if in(i, idx_improved)
                loss_min[i] = loss[i]
                margin_min[i] = margin[i]
                x_best[:, :, :, i] = x_new[:, :, :, i]
            else
                loss_min[i] = loss_min_curr[i]
                margin_min[i] = margin_min_curr[i]
                x_best[:, :, :, i] = x_best_curr[:, :, :, i]
            end
        end

        n_queries[idx_to_fool] .+= 1
    end

    return x_best, n_queries
end