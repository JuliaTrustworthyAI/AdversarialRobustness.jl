using Flux, TaijaData, Random, Distances, AdversarialRobustness, BSON, CUDA, cuDNN
using BSON: @save
include("utils/plot.jl")

# Import MNIST from TaijaData
X, y = load_mnist(10000)
X = (X .+ 1) ./ 2
X = reshape(X, 28, 28, 1, 10000)
y = onehotbatch(y, sort(unique(y)))

# Model architecture
CNN() = Chain(
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10)) |> gpu

# # model_adv_test = CNN();
loss(x, y) = Flux.logitcrossentropy(x, y)
opt = ADAM()

# Standard training
# vanilla_losses = vanilla_train(model, loss, opt, X, y, 5, 32, 0, 9)

# Adv train a dummy model based on the architecture (jaypmorgan to be precise)
# model_adv_pgd3 = CNN()
# adv_losses = adversarial_train(model_adv_pgd3, X, y, 30, 128, 0.3; loss = loss, iterations=10, attack_method = :PGD, opt=opt)
# model_adv_pgd3 = cpu(model_adv_pgd3)
# @save "temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd3.bson" model_adv_pgd3

# Classically trained model
model = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan.bson")[:model]

# FGSM trained 20ep 32bs 0.3ϵ
model_adv = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_advt_test.bson")[:model_adv_test]

# PGD trained 20ep 128bs 0.3ϵ but 0.03 step size and 5 iterations (trained on 10k not 60k)
model_adv_pgd = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd.bson")[:model_adv_pgd]

# Nice PGD 50ep 128bs 0.3e 0.03ss iterations varying from 1 to 10
model_adv_pgd2 = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd2.bson")[:model_adv_pgd2]

# PGD 30ep 128bs 0.3e 0.03ss iterations varying from 1 to 10
model_adv_pgd3 = BSON.load("temp/src/models/MNIST/convnet_jaypmorgan_adv_pgd3.bson")[:model_adv_pgd3]

# Choose specific model to use
# model_to_use = model_adv_pgd
# model_to_use = model
# model_to_use = model_adv
model_to_use = model_adv_pgd2 |> gpu
# model_to_use = model_adv_pgd3 |> gpu

lb = 5
ub = lb + 0

X_try = X[:, :, :, lb:ub] |> gpu
y_try = y[:, lb:ub] |> gpu

# Change to [0, 9] for a target class
target = -1

# Choose attack algorithm: FGSM, PGD, Square and AutoPGD available so far
# x_best_fgsm = FGSM(model_to_use, X_try, y_try; ϵ = 0.2)
x_best_pgd = PGD(model_to_use, X_try, y_try; ϵ = 0.3, step_size=0.01, iterations=20)
# x_best_square, n_queries = SquareAttack(model_to_use, X_try, y_try, 5000; ϵ = 0.3, verbose=true)
# x_best_autopgd, η_list, checkpoints, starts_updated = AutoPGD(model_to_use, X_try, y_try, 100; ϵ = 0.2, target=target)

# attack_to_use = x_best_fgsm
attack_to_use = x_best_pgd
# attack_to_use = x_best_square
# attack_to_use = x_best_autopgd

# println("queries for sq: ", n_queries)

println(extrema(attack_to_use .- X_try))

iw = 1

clean_img = X_try[:, :, :, iw:iw] |> cpu
adv_img = attack_to_use[:, :, :, iw:iw] |> cpu
true_val = y_try[:, iw:iw]

model_to_use = cpu(model_to_use)

clean_pred = model_to_use(clean_img) |> cpu
adv_pred = model_to_use(adv_img) |> cpu

clean_pred_label = (clean_pred |> cpu |> Flux.onecold |> getindex) - 1
adv_pred_label = (adv_pred |> cpu |> Flux.onecold |> getindex) - 1
true_label = (true_val |> cpu |> Flux.onecold |> getindex) - 1

# println("η_list: ", η_list)
# println("checkpoints: ", checkpoints)
# println("starts_updated: ", starts_updated)
# println("n_queries: ", n_queries)
plot_mnist_image(adv_img, clean_img, clean_pred_label, adv_pred_label, true_label)
# difference = adv_img .- clean_img
# plot_mnist_image(difference, clean_img, clean_pred_label, adv_pred_label, true_label)