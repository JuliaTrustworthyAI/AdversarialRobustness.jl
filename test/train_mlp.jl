using Flux
using Flux: DataLoader
using OneHotArrays
using TaijaData

# General setup:
x, y = load_linearly_separable(1000)
xtrain = Float32.(x)
ytrain = onehotbatch(y, sort(unique(y)))
train_set = DataLoader((xtrain, ytrain))
n_hidden = 8
act_fun = relu
model = Chain(
    Dense(size(xtrain, 1), n_hidden, act_fun),
    Dense(n_hidden, n_hidden, act_fun),
    Dense(n_hidden, size(ytrain, 1)),
)
loss(y_hat, y) = Flux.logitcrossentropy(y_hat, y)
rule = Adam()
opt_state = Flux.setup(rule, model)

@testset "Training MLP" begin
    for attack_type in available_attacks

        # Loop over different attack types:
        @testset "Using attack type $(attack_type)" begin

            _model = deepcopy(model)

            for data in train_set
                # Unpack this element (for supervised training):
                input, label = data

                # Attack the input:
                println("Attacking input: $input")
                attack!(attack_type, input, label, _model, loss)
                println("Perturbed input: $input")

                # Calculate the gradient of the objective
                # with respect to the parameters within the model:
                grads = Flux.gradient(_model) do m
                    result = m(input)
                    loss(result, label)
                end

                # Update the parameters so as to reduce the objective,
                # according the chosen optimisation rule:
                Flux.update!(opt_state, _model, grads[1])
            end

            # Tests
            @test true

        end
    end    
end