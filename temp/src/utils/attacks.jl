using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

# White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function FGSM(model, loss, x, y; ϵ = 0.3, min_label=0, max_label=9, clamp_range = (0, 1))
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

# White-box Projected Gradient Descent (PGD) attack by Madry et al. (arxiv.org/abs/1706.06083)
# Code adapted from (github.com/jaypmorgan/Adversarial.jl)
function PGD(model, loss, x, y; ϵ = 0.3, step_size = 0.01, iterations = 40, min_label=0, max_label=9, clamp_range = (0, 1))
    batch_size = size(x)[4]
    x_adv = deepcopy(x)
    for i = 1:batch_size
        x_curr = x_adv[:, :, :, i]
        x_curr = reshape(x_curr, 28, 28, 1, 1)
        x_curr_adv = clamp.(x_curr + (randn(Float32, size(x_curr)...) * Float32(step_size)), clamp_range...); # start from the random point
        δ = chebyshev(x_curr, x_curr_adv)
        iteration = 1; while (δ < ϵ) && iteration <= iterations
            x_curr_adv = FGSM(model, loss, x_curr_adv, y; ϵ = step_size, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
            δ = chebyshev(x_curr, x_curr_adv)
            iteration += 1
        end
        x_adv[:, :, :, i] = x_curr_adv
    end
    return x_adv
end


# Helper functions for the Square Attack

# Selection of p for Square Attack
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

# Margin loss: L(f(x̂), p) = fₚ(x̂) − max(fₖ(x̂)) s.t k≠p
function margin_loss(logits, y, min_label, max_label)
    y = onehotbatch(y, min_label:max_label) 
    preds_correct_class = sum(logits.*y, dims=1)
    diff = preds_correct_class .- logits
    diff[y] .= Inf
    margin = minimum(diff, dims=1)
    return margin
end

# Regular logitcrossentropy without aggregating. Useful for targeted Square Attack
function cross_entropy_loss(logits, y; min_label=0, max_label=9)
    celoss = -sum(onehotbatch(y, min_label:max_label) .* logsoftmax(logits; dims=1); dims=1)
    return celoss
end

# One of the conditions to apply the delta changes according to Andriuschenko et al. but not in advertorch
# Commented out in the actual method
# They describe it as: prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
function check_delta(δ, x_curr_window, x_best_curr_window, center_w, center_h, s, c, i_img, clamp_range = (0, 1))
    δ_window = δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
    clipped_window = clamp.(x_curr_window .+ δ_window, clamp_range...)
    difference = abs.(clipped_window .- x_best_curr_window)
    indices = findall(x -> x < 10^-7, difference)
    return length(indices) == c * s * s
end

# The black-box Square Attack developed by Andriuschenko et al. (https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)
# The only free variable is the budget: iterations
function SquareAttack(model, x, y, iterations, ϵ, p_init, min_label, max_label, verbose; clamp_range = (0, 1))
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
            println("attack successful! All datapoints in this batch are now misclassified")
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

# Helpers for AutoPGD

# Targeted Difference of Logits Ratio (DLR) loss
function targeted_dlr_loss(logits, y, target)
    zy = logits[y, :]
    zt = logits[target, :]
    sorted_logits = sort(logits, dims=1, rev=true)
    zπ1 = sorted_logits[1, :]
    zπ3 = sorted_logits[3, :]
    zπ4 = sorted_logits[4, :]

    return -((zy .- zt) ./ (zπ1 .- ((zπ3 .+ zπ4) ./ 2)))
end

# Condition 1 to halve η and restart from best point
# Returns indices for which the update step has increased f less than ρ * (total update steps since last checkpoint) times
function condition_1(f_list, curr_checkpoint, prev_checkpoint, ρ, n_ex_total)
    update_freqs = zeros(Float64, n_ex_total)
    for i = prev_checkpoint+1:curr_checkpoint-1
        for j in 1:n_ex_total
            if f_list[i][j] > f_list[i-1][j]
                update_freqs[j] += 1
            end
        end
    end

    rhs = ρ * (curr_checkpoint - prev_checkpoint)

    return findall(x -> update_freqs[x] < rhs, 1:n_ex_total)
end

# Condition 2 to halve η and restart from best point
# Returns indices for which no changes happened t 
function condition_2(η_list, f_max_list, curr_ckp_idx, prev_ckp_idx, n_ex_total)
    return findall(x -> (η_list[curr_ckp_idx][x] == η_list[prev_ckp_idx][x]) && (f_max_list[curr_ckp_idx][x] == f_max_list[prev_ckp_idx][x]), 1:n_ex_total)
end


# White-box Auto Projected Gradient Descent: A parameter-free version of PGD (arxiv.org/pdf/2003.01690)
# The only free parameter is the budget: iterations. α and ρ are both set to 0.75 as specified by the authors
# Set a target value to perform the targeted APGD (with DLR loss). Set it to Nothing to perform untargeted APGD (with CE loss)
function AutoPGD(model, x, y, iterations, ϵ, min_label, max_label, verbose; α=0.75, ρ=0.75, clamp_range = (0, 1))
    w, h, c, n_ex_total = size(x)

    # initializing step size
    η = ones(Float64, n_ex_total) .* 2ϵ
    η_list = []
    push!(η_list, η)
    
    # defining checkpoints
    p = [0, 0.22] # period lengths
    checkpoints = [1] # checkpoints for iterations

    while ceil(p[length(p)] * iterations) <= iterations
        last_p = p[length(p)]
        penultimate_p = p[length(p) - 1]
        push!(checkpoints, ceil(last_p * iterations) + 1)
        push!(p, last_p + max(last_p - penultimate_p - 0.03, 0.06))
    end

    x_0 = deepcopy(x)
    x_1 = FGSM(model, cross_entropy_loss, x_0, y; ϵ = η, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
    f_0 = cross_entropy_loss(model(x_0), y; min_label=min_label, max_label=max_label)
    f_1 = cross_entropy_loss(model(x_1), y; min_label=min_label, max_label=max_label)

    f_max = max.(f_0, f_1)
    f_max_list = []
    f_list = []

    push!(f_max_list, f_max)
    push!(f_list, f_max)

    x_max = x_0
    for i in n_ex_total
        if f_max[i] == f_0[i]
            x_max[:, :, :, i] = x_0[:, :, :, i]
        else 
            x_max[:, :, :, i] = x_1[:, :, :, i]
        end
    end

    x_list = []
    push!(x_list, x_0)
    push!(x_list, x_max)

    starts_updated = zeros(Float64, n_ex_total)
    
    for k = 2:iterations
        ηs = deepcopy(η_list[length(η_list)])
        x_k = deepcopy(x_list[k])
        x_k_m_1 = deepcopy(x_list[k-1])
        z_k_p_1 = FGSM(model, cross_entropy_loss, x_k, y; ϵ = ηs, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
        x_k_p_1 = clamp.((x_k + (α .* (z_k_p_1 .- x_k))) + ((1 - α) .* (x_k .- x_k_m_1)), clamp_range...)

        f_x_k_p_1 = cross_entropy_loss(model(x_k_p_1), y; min_label=min_label, max_label=max_label)
        push!(f_list, f_x_k_p_1)

        # Updating maximum loss
        for i in n_ex_total
            if f_x_k_p_1[i] > f_max[i]
                x_max[:, :, :, i] = x_k_p_1[:, :, :, i]
                f_max[i] = f_x_k_p_1[i]
            end
        end

        push!(η_list, ηs)
        push!(x_list, x_k_p_1)
        push!(f_max_list, f_max)

        # Reached checkpoint: check if we need to halve η
        if k in checkpoints
            curr_ckp_idx = findfirst(x -> x==k, checkpoints)
            prev_checkpoint = checkpoints[curr_ckp_idx-1]
            prev_ckp_idx = findfirst(x -> x==prev_checkpoint, checkpoints)

            cond_1 = condition_1(f_list, k, prev_checkpoint, ρ, n_ex_total)
            cond_2 = condition_2(η_list, f_max_list, curr_ckp_idx, prev_ckp_idx, n_ex_total)
            to_update = union(cond_1, cond_2)

            # Indexes for which η is halved and restart point is set to best performing
            for idx in to_update
                ηs[idx] /= 2
                x_k_p_1[:, :, :, idx] = x_max[:, :, :, idx]
                starts_updated[idx] += 1
            end

            η_list[k] = ηs
            x_list[k] = x_k_p_1
        end
    end
    return x_max, η_list, checkpoints, starts_updated
end