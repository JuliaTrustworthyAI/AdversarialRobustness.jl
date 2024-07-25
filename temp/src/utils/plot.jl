using Random, Plots, Flux, Statistics
using Flux: onehotbatch, onecold

function plot_mnist_image(adv_example, clean_example, clean_pred_label, adv_predicted_label, true_label)
    println("Digit predicted by the model for the clean image: ", clean_pred_label)
    println("Digit predicted by the model for the 
    adversarial image: ", adv_predicted_label)
    println("True label: ", true_label)

    image_matrix = reshape(adv_example, 28, 28)
    image_matrix = reverse(permutedims(image_matrix, [2, 1]), dims=1)

    clean_img_matrix = reshape(clean_example, 28, 28)
    clean_img_matrix = reverse(permutedims(clean_img_matrix, [2, 1]), dims=1)
    hms = []
    push!(hms, heatmap(clean_img_matrix, color=:grays, axis=false, title="Clean example: $true_label -> $clean_pred_label"))
    push!(hms, heatmap(image_matrix, color=:grays, axis=false, title="Adv example: $true_label -> $adv_predicted_label"))
    plot(hms..., layout=(1, 2), colorbar=false)
end 

function plot_mnist_clean_and_attack(x, y, attack, model, loss, ϵ; step_size=0.01, iterations=10, attack_method= :FGSM)
    num_datapoints = size(x, 2)
    index = rand(1:num_datapoints)

    clean_example = x[:, index]
    actual = y[index]

    if attack_method == :FGSM
        adversarial_example = attack(model, loss, clean_example, onehotbatch(actual, 0:9); ϵ = ϵ)
    elseif attack_method == :PGD
        adversarial_example = attack(model, loss, clean_example, onehotbatch(actual, 0:9); ϵ=ϵ, step_size=step_size, iterations=iterations)
    else
        error("Unsupported attack method: $attack_method")
    end

    clean_prediction = (model(clean_example) |> Flux.onecold |> getindex) - 1
    adversarial_prediction = (model(adversarial_example) |> Flux.onecold |> getindex) - 1

    plot_mnist_image(adversarial_example, clean_example, clean_prediction, adversarial_prediction, actual)
end

function plot_normal_and_ce(real_image, real_pred, cf_image, cf_pred, actual)
    println("Digit predicted by the model for the clean image: ", real_pred)
    println("Digit predicted by the model for the counterfactual: ", cf_pred)
    println("True label: ", actual)

    image_matrix = reshape(cf_image, 28, 28)
    image_matrix = reverse(permutedims(image_matrix, [2, 1]), dims=1)

    clean_img_matrix = reshape(real_image, 28, 28)
    clean_img_matrix = reverse(permutedims(clean_img_matrix, [2, 1]), dims=1)
    hms = []
    push!(hms, heatmap(clean_img_matrix, color=:grays, axis=false, title="Clean: $real_pred"))
    push!(hms, heatmap(image_matrix, color=:grays, axis=false, title="Counterfactual: $cf_pred"))
    plot(hms..., layout=(1, 2), colorbar=false)
end
