using Flux, TaijaData, Random, Distances, AdversarialRobustness, BSON
include("utils/plot.jl")

# CNN() = Chain(
#     Conv((3, 3), 1=>16, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 16=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 32=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     x -> reshape(x, :, size(x, 4)),
#     Dense(288, 10)) |> gpu

# model = CNN();
# loss(x, y) = logitcrossentropy(x, y)
# opt = ADAM()

# vanilla_losses = vanilla_train(model, loss, opt, X, y, 5, 32, 0, 9)

X, y = load_mnist()
X = (X .+ 1) ./ 2
X = reshape(X, 28, 28, 1, 60000)

model = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan.bson")[:model]

idx = rand(1:1000)

X_try = X[:, :, :, idx]
y_try = y[idx]

x_best_fgsm = FGSM(model, X_try, y_try; ϵ = 0.2)
# x_best_pgd = PGD(model, X_try, y_try; ϵ = 0.3, step_size=0.02, iterations=40)
# x_best_square, n_queries = SquareAttack(model, X_try, y_try, 5000; ϵ = 0.3, verbose=true)
# x_best_autopgd, η_list, checkpoints, starts_updated = AutoPGD(model, X_try, y_try, 100; ϵ = 0.1)

println(extrema(x_best_fgsm .- X_try))

attack_to_use = x_best_fgsm
# attack_to_use = x_best_pgd
# attack_to_use = x_best_square
# attack_to_use = x_best_autopgd

clean_img = X_try[:, :, :, 1:1]
adv_img = attack_to_use[:, :, :, 1:1]

clean_pred = model(clean_img)
adv_pred = model(adv_img)

clean_pred_label = (clean_pred |> Flux.onecold |> getindex) - 1
adv_pred_label = (adv_pred |> Flux.onecold |> getindex) - 1
true_label = y_try

# println("η_list: ", η_list)
# println("checkpoints: ", checkpoints)
# println("starts_updated: ", starts_updated)
# println("n_queries: ", n_queries)
plot_mnist_image(adv_img, clean_img, clean_pred_label, adv_pred_label, true_label)