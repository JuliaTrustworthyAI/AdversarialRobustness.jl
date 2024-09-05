using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

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
    preds_correct_class = sum(logits .* y, dims = 1)
    diff = preds_correct_class .- logits
    diff[y] .= Inf
    margin = minimum(diff, dims = 1)
    return margin
end

# One of the conditions to apply the delta changes according to Andriuschenko et al. but not in advertorch
# Commented out in the actual method
# They describe it as: prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
function check_delta(
    δ,
    x_curr_window,
    x_best_curr_window,
    center_w,
    center_h,
    s,
    c,
    i_img,
    clamp_range = (0, 1),
)
    δ_window = δ[center_w:center_w+s-1, center_h:center_h+s-1, :, i_img]
    clipped_window = clamp.(x_curr_window .+ δ_window, clamp_range...)
    difference = abs.(clipped_window .- x_best_curr_window)
    indices = findall(x -> x < 10^-7, difference)
    return length(indices) == c * s * s
end
