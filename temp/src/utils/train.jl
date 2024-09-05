using Flux, Statistics, ProgressMeter
using Flux.Data: DataLoader
using Flux:
    onehotbatch, onecold, crossentropy, logitcrossentropy, mse, throttle, update!, push!

function vanilla_train(
    model, loss, opt, x_train, y_train, max_epochs, batch_size, label_min, label_max
)
    θ = Flux.params(model)
    vanilla_losses = []
    train_loader = DataLoader((x_train, y_train); batchsize=batch_size, shuffle=true)

    @showprogress for epoch in 1:max_epochs
        println("Epoch: $epoch")
        epoch_loss = 0.0

        for (idx, (x, y)) in enumerate(train_loader)
            # println("Batch: $idx")
            local l
            y_onehot = onehotbatch(y, label_min:label_max)
            grads = Flux.gradient(θ) do
                l = loss(model(x), y_onehot)
            end
            update!(opt, θ, grads)
            epoch_loss += l
        end

        avg_loss = epoch_loss / length(train_loader)

        println("Average loss: $avg_loss")

        push!(vanilla_losses, avg_loss)
    end

    return vanilla_losses
end

function adversarial_train(
    model,
    loss,
    opt,
    x_train,
    y_train,
    epochs,
    batch_size,
    attack,
    label_min,
    label_max,
    ϵ;
    step_size=0.01,
    iterations=10,
    attack_method=:FGSM,
    clamp_range=(0, 1),
)
    adv_losses = []
    θ = Flux.params(model)
    train_loader = DataLoader((x_train, y_train); batchsize=batch_size, shuffle=true)

    initial_ε = 0
    ε_increment = ϵ / epochs

    @showprogress for epoch in 1:epochs
        println("Epoch: $epoch")
        epoch_loss = 0.0
        # ϵ = initial_ε + (epoch - 1) * ε_increment

        for (idx, (x, y)) in enumerate(train_loader)
            y_onehot = onehotbatch(y, label_min:label_max)
            x_adv = zeros(size(x))

            if attack_method == :FGSM
                x_adv = attack(model, loss, x, y_onehot; ϵ=ϵ, clamp_range=clamp_range)
            elseif attack_method == :PGD
                x_adv = attack(
                    model,
                    loss,
                    x,
                    y_onehot;
                    ϵ=ϵ,
                    step_size=step_size,
                    iterations=iterations,
                    clamp_range=clamp_range,
                )
            else
                error("Unsupported attack method: $attack_method")
            end

            l_adv = 0.0
            l_nat = 0.0

            grads = Flux.gradient(θ) do
                l_adv = loss(model(x_adv), y_onehot)
                l_nat = loss(model(x), y_onehot)
                return l_adv + l_nat
            end

            update!(opt, θ, grads)
            epoch_loss += (l_adv + l_nat)
        end

        avg_loss = epoch_loss / length(train_loader)

        println("Average loss: $avg_loss")

        push!(adv_losses, avg_loss)
    end

    return adv_losses
end
