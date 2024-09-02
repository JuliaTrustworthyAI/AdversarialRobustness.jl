include("common_utils.jl")

include("fgsm/fgsm.jl")
include("pgd/pgd.jl")
include("square/square_attack.jl")
include("autopgd/autopgd.jl")

const available_attacks = [
    FGSM,
    PGD,
    AutoPGD
]

"""
    attack(type::Function, x, y, model; kwargs...)

Attacks the `model` on input `x` with label `y` using the attack `type`.
"""
function attack(type::Function, x, y, model, loss; kwargs...)
    return type(x, y, model; loss=loss, kwargs...)
end

"""
    attack!(x, y, model, type::Function; kwargs...)

Attacks the `model` on input `x` with label `y` using the attack `type` in-place.
"""
function attack!(type::Function, x, y, model, loss; kwargs...)
    return attack(type, x, y, model, loss; kwargs...)
end