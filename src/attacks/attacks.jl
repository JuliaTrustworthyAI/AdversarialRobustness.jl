include("common_utils.jl")

include("fgsm/fgsm.jl")
include("pgd/pgd.jl")
include("square/square_attack.jl")
include("autopgd/autopgd.jl")

const available_attacks = [
    FGSM,
    PGD,
    AutoPGD,
    # SquareAttack,
]

"""
    attack(type::Function, x, y, model; kwargs...)

Attacks the `model` on input `x` with label `y` using the attack `type`.
"""
function attack(type::Function, x, y, model, loss; kwargs...)
    x = (xadv -> convert.(eltype(x), xadv))(type(model, x, y; loss=loss, kwargs...))
    return x
end

"""
     attack!(type::Function, x, y, model, loss; kwargs...)

Attacks the `model` on input `x` with label `y` using the attack `type` in-place (i.e. argument `x` is mutated).
"""
function attack!(type::Function, x, y, model, loss; kwargs...)
    x[:] = attack(type, x, y, model, loss; kwargs...)
    return x
end
