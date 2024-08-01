using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

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
function condition_1(f_list, curr_checkpoint, prev_checkpoint, ρ)
    update_freqs = 0
    for i = prev_checkpoint+1:curr_checkpoint-1
        if f_list[i] > f_list[i-1]
            update_freqs += 1
        end
    end

    rhs = ρ * (curr_checkpoint - prev_checkpoint)

    return update_freqs < rhs
end

# Condition 2 to halve η and restart from best point
# Returns indices for which no changes happened t 
function condition_2(η_list, f_max_list, curr_ckp_idx, prev_ckp_idx)
    return (η_list[curr_ckp_idx] == η_list[prev_ckp_idx]) && (f_max_list[curr_ckp_idx] == f_max_list[prev_ckp_idx])
end