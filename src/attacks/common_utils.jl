using Distributions
using Random
using Flux, Statistics, Distances
using Flux: onehotbatch, onecold

function cross_entropy_loss(logits, y; min_label = 0, max_label = 9)
    celoss =
        -sum(onehotbatch(y, min_label:max_label) .* logsoftmax(logits; dims = 1); dims = 1)
    return celoss
end
