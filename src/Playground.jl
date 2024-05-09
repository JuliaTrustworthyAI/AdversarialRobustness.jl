# println("Hello World!")
using BSON
using BSON: @save, @load
using Random
using Flux, Statistics, ProgressMeter, Plots, TaijaData, Distances
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy, throttle, update!, push!
using Base.Iterators: repeated, partition

x_train, y_train = TaijaData.load_mnist()
x_test, y_test = TaijaData.load_mnist_test()

x_train = reshape(x_train, (28, 28, 1, 60000))
x_test = reshape(x_test, (28, 28, 1, 10000))

x_first_10k = x_train[:, :, :, 1:10000]
y_first_10k = y_train[1:10000]

m = BSON.load("play_models/cnn_10k_5_32.bson")[:m2] 
θ = Flux.params(m)
loss(x, y) = crossentropy(m(x), y)
acc(x, y) = mean(m(x) .== Flux.onecold(y))
opt = ADAM()

function FGSM(model, loss, x, y; ϵ = 0.5, clamp_range = (-1, 1))
    J = gradient(() -> loss(x, y), Flux.params([x]))
    x_adv = clamp.(x .+ (Float32(ϵ) * sign.(J[x])), clamp_range...)
    return x_adv
end

function PGD(model, loss, x, y; ϵ = 10, step_size = 0.01, iters = 100, clamp_range = (-1, 1))
    x_adv = clamp.(x + (randn(Float32, size(x)...) * Float32(step_size)), clamp_range...); # start from the random point
    δ = Distances.chebyshev(x, x_adv)
    iter = 1; while (δ < ϵ) && iter <= iters
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iter += 1
    end
    return x_adv
end

function plot_image(adv_example, clean_example, clean_pred_label, predicted_label, true_label)
    println("Digit predicted by the model for the clean image: ", clean_pred_label)
    println("Digit predicted by the model for the adversarial image: ", predicted_label)
    println("True label: ", true_label)

    image_matrix = reshape(adv_example, 28, 28)
    image_matrix = reverse(permutedims(image_matrix, [2, 1]), dims=1)

    clean_img_matrix = reshape(clean_example, 28, 28)
    clean_img_matrix = reverse(permutedims(clean_img_matrix, [2, 1]), dims=1)
    hms = []
    push!(hms, heatmap(image_matrix, color=:grays, axis=false, title="$true_label -> $predicted_label"))
    push!(hms, heatmap(clean_img_matrix, color=:grays, axis=false, title="$true_label -> $clean_pred_label"))
    plot(hms..., layout=(1, 2), colorbar=false)

end


# Demo for adversarial examples for MNIST
# Dummy initialization
adv_pred = 2
real_pred = 2
attempts_taken = 0
adversarial_example = 0
real_example = 0
true_label = -1
index = -1

while adv_pred == real_pred
    index = rand(1:60000)
    real_example = reshape(x_train[:, :, :, index], 28, 28, 1, 1)
    true_label = y_train[index]
    adversarial_example = FGSM(m, loss, real_example, true_label)
    # adversarial_example = PGD(m, loss, real_example, true_label)
    adv_pred = m(adversarial_example) |> Flux.onecold |> getindex
    real_pred = m(real_example) |> Flux.onecold |> getindex
    attempts_taken += 1
end

println("attempts taken to find adversarial example that classifies different: ", attempts_taken)
println("index identified: ", index)
plot_image(adversarial_example, real_example, real_pred - 1, adv_pred - 1, true_label)

































# CNN() = Chain(
#     Conv((3, 3), 1=>16, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 16=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 32=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     x -> reshape(x, :, size(x, 4)),
#     Dense(288, 10),
#     softmax
# )

# m = CNN()
# θ = Flux.params(m)
# loss(x, y) = crossentropy(m(x), y)
# acc(x, y) = mean(m(x) .== Flux.onecold(y))
# opt = ADAM()

# epochs = 10
# losses = []

# train_loader = DataLoader((x_first_10k , y_first_10k), batchsize=128, shuffle=true)

# @showprogress for epoch in 1:epochs
#     println("Epoch: $epoch")
#     epoch_loss = 0.0

#     for (idx, (x, y)) in enumerate(train_loader)
#         println("Batch: $idx")
#         local l;
#         y_onehot = onehotbatch(y, 0:9)
#         grads = Flux.gradient(θ) do 
#             l = loss(x, y_onehot)    
#         end

#         update!(opt, θ, grads)

#         epoch_loss += l

#     end

#     avg_loss = epoch_loss/length(train_loader)

#     println("Average loss: $avg_loss")

#     push!(losses, avg_loss)
# end

# noisy = rand(Float32, 2, 1000)
# truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]

# model = Chain(
#     Dense(2 => 3, tanh), # First layer, 2 input neurons and 3 output neurons with tanh activation
#     BatchNorm(3),  # Batch normalization?
#     Dense(3 => 2), # Second layer, 3 in 2 out
#     softmax # Softmax duh
# )

# out1 = model(noisy) # forward pass the input noise through the model to generate output

# target = Flux.onehotbatch(truth, [true, false]) # TODO: WTF is this
# loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true) # Loads data in x-y way?

# optim = Flux.setup(Flux.Adam(0.01), model) # Sets up optimizer for model

# losses = []

# @showprogress for epoch in 1:1_000
#     for (x, y) in loader
#         loss, grads = Flux.withgradient(model) do m
#             y_hat = m(x)
#             Flux.crossentropy(y_hat, y)
#         end
#         Flux.update!(optim, model, grads[1])
#         push!(losses, loss)
#     end
# end

# optim

# out2 = model(noisy)

# mean((out2[1,:] .> 0.5) .== truth)

# p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
# p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=out1[1,:], title="Untrained network", label="", clims=(0,1))
# p_done = scatter(noisy[1,:], noisy[2,:], zcolor=out2[1,:], title="Trained network", legend=false)

# plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))

# plot(losses; xaxis=(:log10, "iteration"),
#     yaxis="loss", label="per batch")
# n = length(loader)
# plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
#     label="epoch mean", dpi=200)