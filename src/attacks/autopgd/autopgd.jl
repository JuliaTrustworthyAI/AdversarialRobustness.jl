using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

include("utils.jl")
# include("../common_utils.jl")

# White-box Auto Projected Gradient Descent: A parameter-free version of PGD (arxiv.org/pdf/2003.01690)
# The only free parameter is the budget: iterations. α and ρ are both set to 0.75 as specified by the authors
function AutoPGD(model, x, y, iterations; ϵ=0.3, min_label=0, max_label=9, verbose=false, α=0.75, ρ=0.75, clamp_range = (0, 1))
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
    x_1 = FGSM(model, x_0, y; loss=cross_entropy_loss, ϵ = η, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
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
        z_k_p_1 = FGSM(model, x_k, y; loss=cross_entropy_loss, ϵ = ηs, min_label=min_label, max_label=max_label, clamp_range = clamp_range)
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