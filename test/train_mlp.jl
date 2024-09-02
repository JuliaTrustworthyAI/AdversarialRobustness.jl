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

@testset "Training MLP" begin
    
end