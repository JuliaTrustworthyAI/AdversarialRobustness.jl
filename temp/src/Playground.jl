using Flux, TaijaData, Random, Distances, AdversarialRobustness, BSON
using BSON: @save
include("utils/plot.jl")

# Import MNIST from TaijaData
X, y = load_mnist(10000)
X = (X .+ 1) ./ 2
X = reshape(X, 28, 28, 1, 10000)

# Model architecture
# CNN() = Chain(
#     Conv((3, 3), 1=>16, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 16=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 32=>32, pad=(1,1), relu),
#     MaxPool((2,2)),
#     x -> reshape(x, :, size(x, 4)),
#     Dense(288, 10)) |> gpu

# model_adv_test = CNN();
# loss(x, y) = Flux.logitcrossentropy(x, y)
# opt = ADAM()

# Standard training
# vanilla_losses = vanilla_train(model, loss, opt, X, y, 5, 32, 0, 9)

# Adv train a dummy model based on the architecture (jaypmorgan to be precise)
# model_adv_pgd = CNN()
# adv_losses = adversarial_train(model_adv_pgd, X, y, 20, 128, 0.3; loss = loss, attack_method = :PGD, opt=opt)
# @save "temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd.bson" model_adv_pgd

# Classically trained model
model = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan.bson")[:model]

# FGSM trained 20ep 32bs 0.3ϵ
model_adv = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_advt_test.bson")[:model_adv_test]

# PGD trained 20ep 128bs 0.3ϵ but 0.03 step size and 5 iterations (trained on 10k not 60k)
model_adv_pgd = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd.bson")[:model_adv_pgd]

# Choose specific model to use
# model_to_use = model_adv_pgd
# model_to_use = model
model_to_use = model_adv

idx = rand(1:10000)

X_try = X[:, :, :, idx]
y_try = y[idx]

# Change to [0, 9] for a target class
target = -1

# Choose attack algorithm: FGSM, PGD, Square and AutoPGD available so far
# x_best_fgsm = FGSM(model_to_use, X_try, y_try; ϵ = 0.3)
x_best_pgd = PGD(model_to_use, X_try, y_try; ϵ = 0.3, step_size=0.05, iterations=5)
# x_best_square, n_queries = SquareAttack(model_to_use, X_try, y_try, 5000; ϵ = 0.3, verbose=true)
# x_best_autopgd, η_list, checkpoints, starts_updated = AutoPGD(model_to_use, X_try, y_try, 100; ϵ = 0.2, target=target)

# attack_to_use = x_best_fgsm
attack_to_use = x_best_pgd
# attack_to_use = x_best_square
# attack_to_use = x_best_autopgd

println(extrema(attack_to_use .- X_try))

clean_img = X_try[:, :, :, 1:1]
adv_img = attack_to_use[:, :, :, 1:1]

clean_pred = model_to_use(clean_img)
adv_pred = model_to_use(adv_img)

clean_pred_label = (clean_pred |> Flux.onecold |> getindex) - 1
adv_pred_label = (adv_pred |> Flux.onecold |> getindex) - 1
true_label = y_try

# println("η_list: ", η_list)
# println("checkpoints: ", checkpoints)
# println("starts_updated: ", starts_updated)
# println("n_queries: ", n_queries)
plot_mnist_image(adv_img, clean_img, clean_pred_label, adv_pred_label, true_label)
# difference = adv_img .- clean_img
# plot_mnist_image(difference, clean_img, clean_pred_label, adv_pred_label, true_label)