using Flux, Statistics, ProgressMeter
using Flux.Data: DataLoader
using Flux:
    onehotbatch, onecold, crossentropy, logitcrossentropy, mse, throttle, update!, push!

function vanilla_train(
    model,
    x_train,
    y_train,
    max_epochs,
    batch_size;
    loss = logitcrossentropy,
    opt = Adam,
)
    θ = Flux.params(model)
    vanilla_losses = []
    train_loader = DataLoader((x_train, y_train), batchsize = batch_size, shuffle = true) |> gpu

    @showprogress for epoch = 1:max_epochs
        println("Epoch: $epoch")
        epoch_loss = 0.0

        for (idx, (x, y)) in enumerate(train_loader)
            # println("Batch: $idx")
            local l
            # y_onehot = onehotbatch(y, label_min:label_max)
            grads = Flux.gradient(θ) do
                l = loss(model(x), y)
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
    x_train,
    y_train,
    epochs,
    batch_size,
    ϵ;
    loss = logitcrossentropy,
    opt = Adam,
    step_size = 0.01,
    iterations = 10,
    attack_method = :FGSM,
    clamp_range = (0, 1),
)
    adv_losses = []
    θ = Flux.params(model)
    train_loader = DataLoader((x_train, y_train), batchsize = batch_size, shuffle = true) |> gpu

    iter = ceil(iterations/epochs)
    iter_val = iterations/epochs

    @showprogress for epoch = 1:epochs
        println("Epoch: $epoch")
        println("number of iterations for PGD: ", iter)
        epoch_loss = 0.0

        for (idx, (x, y)) in enumerate(train_loader)
            if idx % 100 == 0
                println("batch ", idx)
            end
            # y_onehot = onehotbatch(y, min_label:max_label)
            x_adv = zeros(size(x))

            if attack_method == :FGSM
                x_adv = FGSM(
                    model,
                    x,
                    y;
                    loss = loss,
                    ϵ = ϵ,
                    clamp_range = clamp_range,
                )
            elseif attack_method == :PGD
                x_adv = PGD(
                    model,
                    x,
                    y;
                    loss = loss,
                    ϵ = ϵ,
                    clamp_range = clamp_range,
                    step_size = step_size,
                    iterations = iter
                )
            else
                error("Unsupported attack method: $attack_method")
            end

            l_adv = 0.0

            grads = Flux.gradient(θ) do
                l_adv = loss(model(x_adv), y)
                return l_adv
            end

            update!(opt, θ, grads)
            epoch_loss += (l_adv)

        end

        avg_loss = epoch_loss / length(train_loader)

        println("Average loss: $avg_loss")

        push!(adv_losses, avg_loss)
        
        iter_val += iterations/epochs
        iter = ceil(iter_val)
    end

    return adv_losses
end
