module AdversarialRobustness

# Element-wise cross-entropy loss
include("attacks/common_utils.jl")
export cross_entropy_loss

# Attack algorithms supported
include("attacks/autopgd/autopgd.jl")
export AutoPGD

include("attacks/square/square_attack.jl")
export SquareAttack

include("attacks/fgsm/fgsm.jl")
export FGSM

include("attacks/pgd/pgd.jl")
export PGD

# Adversarial training
include("training/adversarial_training.jl")
export vanilla_train, adversarial_train

end
