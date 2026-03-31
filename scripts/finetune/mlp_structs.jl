struct MLPModel{P,H}
    pretrained::P
    head::H
end

Flux.@layer MLPModel

function MLPModel(; input_size::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = Flux.Chain() 
    head = Flux.Chain(
        Flux.Dense(input_size => hidden_dim, gelu),
        Flux.LayerNorm(hidden_dim),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => hidden_dim, gelu),
        Flux.LayerNorm(hidden_dim),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications)
    )
    return MLPModel(pretrained, head)
end

function (m::MLPModel)(input)
    x = Float32.(input) 
    return m.head(x)
end