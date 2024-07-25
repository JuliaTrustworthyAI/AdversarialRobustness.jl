using AdversarialRobustness
using Documenter

DocMeta.setdocmeta!(AdversarialRobustness, :DocTestSetup, :(using AdversarialRobustness); recursive=true)

makedocs(;
    modules=[AdversarialRobustness],
    authors="Rithik Appachi Senthilkumar",
    sitename="AdversarialRobustness.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaTrustworthyAI.github.io/AdversarialRobustness.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/AdversarialRobustness.jl",
    devbranch="master",
)
