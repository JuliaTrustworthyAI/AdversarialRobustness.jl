module AdversarialRobustness

# Attack algorithms supported
include("attacks/attacks.jl")
export AutoPGD
export SquareAttack
export FGSM
export PGD
export attack, attack!, available_attacks

# Adversarial training
include("training/adversarial_training.jl")
export vanilla_train, adversarial_train

end
