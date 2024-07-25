using Flux, Statistics
using Flux: onecold

function evaluate_model(x_test, y_test, label_min, label_max, model, loss, attack, ϵ; step_size=0.01, iterations=10, attack_method= :FGSM,clamp_range=(0, 1))
    num_datapoints = size(x_test, 2)

    adversary_changed = 0
    adversary_wrong = 0
    clean_wrong = 0
    clean_right_but_adv_wrong = 0

    for index = 1:num_datapoints
        clean_example = x_test[:, index]
        actual = y_test[index]

        clean_pred = (softmax(model(clean_example)) |> Flux.onecold |> getindex) - 1

        adversarial_example = clean_example

        if attack_method == :FGSM
            adversarial_example = attack(model, loss, clean_example, onehotbatch(actual, label_min:label_max); ϵ = ϵ, clamp_range = clamp_range)
        elseif attack_method == :PGD
            adversarial_example = attack(model, loss, clean_example, onehotbatch(actual, label_min:label_max); ϵ = ϵ, step_size = step_size, iterations = iterations, clamp_range = clamp_range)
        else 
            error("Unsupported attack method: $attack_method")
        end

        adv_pred = (softmax(model(adversarial_example)) |> Flux.onecold |> getindex) - 1

        if adv_pred != clean_pred
            adversary_changed += 1
        end
    
        if adv_pred != actual
            adversary_wrong += 1
        end
    
        if clean_pred != actual
            clean_wrong += 1
        end
    
        if clean_pred == actual && adv_pred != actual
            clean_right_but_adv_wrong += 1
        end
    end

    overview = Dict()

    overview["clean_accuracy"] = (1 - clean_wrong/num_datapoints)
    overview["adversarial_accuracy"] = (1 - adversary_wrong/num_datapoints)
    overview["adversary_changed"] = adversary_changed
    overview["adversary_forced_error"] = clean_right_but_adv_wrong

    return overview
end
