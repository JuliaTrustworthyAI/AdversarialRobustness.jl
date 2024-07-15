using Distributions
using Random
using Flux, Statistics, ProgressMeter, Distances
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy, logitcrossentropy, mse, throttle, update!, push!

function FGSM(model, loss, x, y; ϵ = 0.3, clamp_range = (0, 1))
    grads = gradient(x -> loss(model(x), y), x)[1]
    x_adv = clamp.(x + (Float32(ϵ) * sign.(grads)), clamp_range...)
    return x_adv
end

function PGD(model, loss, x, y; ϵ = 0.3, step_size = 0.01, iterations = 40, clamp_range = (0, 1))
    x_adv = clamp.(x + (randn(Float32, size(x)...) * Float32(step_size)), clamp_range...); # start from the random point
    δ = Distances.chebyshev(x, x_adv)
    iteration = 1; while (δ < ϵ) && iteration <= iterations
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iteration += 1
    end
    return x_adv
end

# Selection of p for SquareAttack
function p_selection(p_init, it, n_iters)
    scaled_it = Int(round(it / n_iters * 10000))

    if 10 < scaled_it && scaled_it <= 50
        p = p_init / 2
    elseif 50 < scaled_it && scaled_it <= 200
        p = p_init / 4
    elseif 200 < scaled_it && scaled_it <= 500
        p = p_init / 8
    elseif 500 < scaled_it && scaled_it <= 1000
        p = p_init / 16
    elseif 1000 < scaled_it && scaled_it <= 2000
        p = p_init / 32
    elseif 2000 < scaled_it && scaled_it <= 4000
        p = p_init / 64
    elseif 4000 < scaled_it && scaled_it <= 6000
        p = p_init / 128
    elseif 6000 < scaled_it && scaled_it <= 8000
        p = p_init / 256
    elseif 8000 < scaled_it && scaled_it <= 10000
        p = p_init / 512
    else
        p = p_init
    end

    return p
end

function margin_loss(logits, y)
    y = onehotbatch(y, 0:9) 
    preds_correct_class = sum(logits.*y, dims=1)
    diff = preds_correct_class .- logits
    diff[y] .= Inf
    margin = minimum(diff, dims=1)
    return margin
end

function cross_entropy_loss(logits, y)
    return -sum(onehotbatch(y, 0:9) .* logsoftmax(logits; dims=1); dims=1)
end

function check_delta(δ, x_curr_window, x_best_curr_window, center_w, center_h, s, c, i_img, min_val, max_val)
    δ_window = δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
    clipped_window = clamp.(x_curr_window .+ δ_window, min_val, max_val)
    difference = abs.(clipped_window .- x_best_curr_window)
    indices = findall(x -> x < 10^-7, difference)
    return length(indices) == c * s * s
end

function SquareAttack(model, x, y, iterations, p_init, ϵ, min_val, max_val, verbose)
    Random.seed!(1)
    w, h, c, n_ex_total = size(x)
    n_features = c*h*w

    # Initialization (stripes of +/-ϵ)
    init_δ = rand(w, 1, c, n_ex_total) .|> x -> x < 0.5 ? -ϵ : ϵ
    init_δ_extended = repeat(init_δ, 1, h, 1, 1)
    x_best = clamp.((init_δ_extended + x), min_val, max_val)

    copylol = deepcopy(x_best)
    x_topass = reshape(copylol, w*h, n_ex_total) # TODO: change soon, currently passes into direct models with no conv layers

    logits = model(x_topass)
    loss_min = cross_entropy_loss(logits, y)
    margin_min = margin_loss(logits, y)
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
            println("attack successful!")
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

            # x_curr_window = x_curr[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
            # x_best_curr_window = x_best_curr[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]

            random_choices = rand([0, 1], c)
            values = (random_choices .* 2 .- 1) .* ϵ 

            δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img] .= values 

            # while check_delta(δ, x_curr_window, x_best_curr_window, center_w, center_h, s, c, i_img, min_val, max_val)
            #     while_entries += 1
            #     random_choices = rand([0, 1], c)
            #     values = (random_choices .* 2 .- 1) .* ϵ 
            #     δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img] .= values 
            # end
        end

        x_new = clamp.(x_curr + δ, min_val, max_val)
        copylol = deepcopy(x_new)
        x_topass = reshape(copylol, w*h, n_ex_total) # TODO: change soon, currently passes into direct models with no conv layers

        logits = model(x_topass)
        loss = cross_entropy_loss(logits, y_curr)
        margin = margin_loss(logits, y_curr)

        idx_improved = findall(x -> loss[x] < loss_min_curr[x], 1:n_ex_total)
        times_it_actually_improved += length(idx_improved)

        if length(idx_improved) > 0 && verbose
            println("Square modification caused a better loss here!")
            println("iteration: ", iteration)
            println("index improved by Square: ", idx_improved)
            println("side length to achieve this: ", s)
        end

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

    if verbose
        println("times_it_actually_improved: ", times_it_actually_improved)
    end

    return x_best, n_queries
end