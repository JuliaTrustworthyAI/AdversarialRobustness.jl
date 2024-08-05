using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

include("utils.jl")

# White-box Auto Projected Gradient Descent: A parameter-free version of PGD (arxiv.org/pdf/2003.01690)
# The only free parameter is the budget: iterations. α and ρ are both set to 0.75 as specified by the authors
function AutoPGD(model, x, y, iterations; ϵ=0.3, target=-1, min_label=0, max_label=9, verbose=false, α=0.75, ρ=0.75, clamp_range = (0, 1))
    w, h, c = size(x)

    # initializing step size
    η = 2ϵ
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
    x_1 = clamp.(FGSM(model, x_0, y; loss=cross_entropy_loss, ϵ = η, min_label=min_label, max_label=max_label, clamp_range = (x_0 .- ϵ, x_0 .+ ϵ)), clamp_range...)

    topass_x_0 = deepcopy(x)
    topass_x_0 = reshape(topass_x_0, w, h, c, 1)
    topass_x_1 = deepcopy(x_1)
    topass_x_1 = reshape(topass_x_1, w, h, c, 1)

    logits_0 = model(topass_x_0)
    logits_1 = model(topass_x_1)

    f_0 = logitcrossentropy(logits_0, onehotbatch(y, min_label:max_label))
    f_1 = logitcrossentropy(logits_1, onehotbatch(y, min_label:max_label))

    if target > -1
        f_0 = targeted_dlr_loss(logits_0, y, target)
        f_1 = targeted_dlr_loss(logits_1, y, target)
    end

    f_max = max(f_0, f_1)
    f_max_list = []
    f_list = []

    push!(f_max_list, f_max)
    push!(f_list, f_max)

    x_max = x_0
    if f_max == f_0
        x_max = x_1
    end

    x_list = []
    push!(x_list, x_0)
    push!(x_list, x_max)

    starts_updated = 0
    
    for k = 2:iterations
        η = deepcopy(η_list[length(η_list)])
        x_k = deepcopy(x_list[k])
        x_k_m_1 = deepcopy(x_list[k-1])
        z_k_p_1 = clamp.(FGSM(model, x_k, y; loss=cross_entropy_loss, ϵ = η, min_label=min_label, max_label=max_label, clamp_range = (x_0 .- ϵ, x_0 .+ ϵ)), clamp_range...)
        x_k_p_1 = clamp.(clamp.((x_k + (α .* (z_k_p_1 .- x_k))) + ((1 - α) .* (x_k .- x_k_m_1)), x_0 .- ϵ, x_0 .+ ϵ), clamp_range...)

        topass_xkp1 = deepcopy(x_k_p_1)
        topass_xkp1 = reshape(topass_xkp1, w, h, c, 1)

        logits_xkp1 = model(topass_xkp1)
        
        f_x_k_p_1 = logitcrossentropy(logits_xkp1, onehotbatch(y, min_label:max_label))

        if target > -1
            f_x_k_p_1 = targeted_dlr_loss(logits_xkp1, y, target)
        end
        
        push!(f_list, f_x_k_p_1)

        # Updating maximum loss
        if f_x_k_p_1 > f_max
            x_max = x_k_p_1
            f_max = f_x_k_p_1
        end

        push!(η_list, η)
        push!(x_list, x_k_p_1)
        push!(f_max_list, f_max)

        # Reached checkpoint: check if we need to halve η
        if k in checkpoints
            curr_ckp_idx = findfirst(x -> x==k, checkpoints)
            prev_checkpoint = checkpoints[curr_ckp_idx-1]
            prev_ckp_idx = findfirst(x -> x==prev_checkpoint, checkpoints)

            cond_1 = condition_1(f_list, k, prev_checkpoint, ρ)
            cond_2 = condition_2(η_list, f_max_list, curr_ckp_idx, prev_ckp_idx)

            # Indexes for which η is halved and restart point is set to best performing
            if cond_1 || cond_2
                η /= 2
                x_k_p_1 = x_max
                starts_updated += 1
            end

            η_list[k] = η
            x_list[k] = x_k_p_1
        end
    end
    return x_max, η_list, checkpoints, starts_updated
end