Base.@kwdef mutable struct Config
    dataset::String = "trt"
    data_path::String = "data/lincs_trt_data.jld2"
    model_dir::String = ""

    level::String = "lvl2"
    batch_size::Int = 128
    n_epochs::Int = 1
    pt1_epochs::Int = 1
    pt2_epochs::Int = 1
    cp_freq::Union{Int,Nothing} = nothing

    embed_dim::Int = 128
    drop_prob::Float64 = 0.15
    lr::Float64 = 0.001
    mask_ratio::Float64 = 0.15
    hidden_dim::Int = 256
    n_heads::Int = 2
    n_layers::Int = 4
    modeltype::String = "mlp"
    pca_mode::Symbol = :none

    additional_notes::String = "test"
end
config = Config()