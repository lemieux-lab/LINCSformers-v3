# finetuning w/ same model, diff inputs

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, cuDNN, StatsBase

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")
include("train.jl")

# settings

level = "lvl1"
modeltype = "rtf"
epochs = 100
additional_notes = "even longer test w/o lr decrease"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversampling = level == "lvl2"

include("$(modeltype)_structs.jl")

if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-24_02-55"
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-31_22-35"
elseif modeltype == "v2"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v2/2026-03-31_08-46"
else
    error("check ur modeltype!!! or add etf configurations")
end

gpu_info = CUDA.name(device())
if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
    batch_size = 64
elseif gpu_info == "Tesla V100-SXM2-32GB"
    batch_size = 128
else
    error("check ur gpu!!!")
end

start_time = now()
CUDA.device!(0)

data = load(data_path)["filtered_data"]

X = data.expr 

if level == "lvl1"
    y = data.inst.cell_mfc_name 
    idx = 1:size(X, 2)
elseif level == "lvl2"
    y = data.inst.pert_id 

    cls = countmap(data.inst.pert_id)
    fcls = filter(cls) do p
        p.second > 500 && p.second < 20000
    end |> Dict

    labels = collect(keys(fcls)) |> Set
    idx = findall(l -> l in labels, y)
    X = X[:, idx] 
    y = y[idx]
else
    error("level undefined so y labels are undefined :(")
end

labels = unique(y)
n_classifications = length(labels)
ids = Dict(l => i for (i, l) in enumerate(labels))
y_ids = [ids[l] for l in y]
y_oh = Flux.onehotbatch(y_ids, 1:n_classifications)

n_genes = size(X, 1) 
n_classes_pt = n_genes
n_features_pt = n_classes_pt + 1

gene_medians = vec(median(X, dims=2)) .+ 1e-10
X_ranked = rank_genes(X, gene_medians)

if modeltype == "mlp"
    X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
    y_train = y_oh[:, train_indices]
    y_test = y_oh[:, test_indices]
else
    pt_indices = load(joinpath(dir, "indices.jld2"))
    pt_train_indices = pt_indices["train_indices"]
    pt_test_indices  = pt_indices["test_indices"]

    idx_dict = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(idx))
    train_indices = [idx_dict[i] for i in pt_train_indices if haskey(idx_dict, i)]
    test_indices  = [idx_dict[i] for i in pt_test_indices if haskey(idx_dict, i)]

    X_train = X[:, train_indices]
    X_test  = X[:, test_indices]
    y_train = y_oh[:, train_indices]
    y_test  = y_oh[:, test_indices]
end

if use_pca
    raw_train_pt = data.expr[:, pt_train_indices]
    pca_train_norm = fit(PCA, Float32.(raw_train_pt); maxoutdim=embed_dim)
    raw_train = X[:, train_indices]
    raw_test = X[:, test_indices]
end

if use_oversampling
    y_train_labels = Flux.onecold(cpu(y_train))
    cidx_dict = Dict{Int, Vector{Int}}()
    for (idx, class) in enumerate(y_train_labels)
        push!(get!(cidx_dict, class, Int[]), idx)
    end
    cs = collect(keys(cidx_dict))
else
    cidx_dict, cs = nothing, nothing
end

function oversample_batch(class_dict, classes, b_size)
    return [rand(class_dict[rand(classes)]) for _ in 1:b_size]
end

if modeltype == "mlp"
    train_input = Float32.(X_train)
    test_input = Float32.(X_test)
    ft_model = Flux.Chain(
        Flux.Dense(n_genes => hidden_dim, gelu),
        Flux.LayerNorm(hidden_dim),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications)
        ) |> gpu
else
    state = load("$dir/model_state.jld2")["model_state"]
    general_model = (
        input_size=n_features_pt,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_classes=n_classes_pt,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        dropout_prob=drop_prob)

    if modeltype == "rtf"
        pt_model = Model(; general_model...) |> gpu
    elseif modeltype == "v1"
        pt_model = Model(; general_model..., pca_dim=embed_dim, use_pca_proj=false) |> gpu
    elseif modeltype == "v2"
        pt_model = Model(; general_model..., pca_dim=embed_dim) |> gpu
    end
    Flux.loadmodel!(pt_model, state)
    Flux.testmode!(pt_model)

    function get_pooled(m, x, x_pca=nothing, m_type=modeltype)
        embedded = m.embedding(x)
        
        if m_type == "rtf"
            encoded = m.pos_encoder(embedded)
            
        elseif m_type == "v1"
            processed_pca = m.use_pca_proj ? m.pca_proj(x_pca) : x_pca
            pca_reshaped = reshape(processed_pca, size(processed_pca, 1), 1, size(processed_pca, 2))
            combined = cat(pca_reshaped, embedded, dims=2)
            encoded = m.pos_encoder(combined)
            
        elseif m_type == "v2"
            pca_embedded = m.pca_proj(x_pca)
            pca_normed = m.pca_norm(pca_embedded)
            pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2))
            combined = embedded .+ pca_reshaped
            encoded = m.pos_encoder(combined)
        end
        
        encoded_dropped = m.pos_dropout(encoded)
        transformed = m.transformer(encoded_dropped)
        return dropdims(mean(transformed, dims=2), dims=2)
    end

    train_embeds = []
    for i in 1:batch_size:size(X_train, 2)
        b_idx = i:min(i+batch_size-1, size(X_train, 2))
        x_batch = gpu(Int32.(X_ranked[:, b_idx]))
        
        if use_pca
            x_pca_batch = gpu(Float32.(MultivariateStats.predict(pca_train_norm, Float32.(raw_train[:, b_idx]))))
            push!(train_embeds, cpu(get_pooled(pt_model, x_batch, x_pca_batch, modeltype)))
        else
            push!(train_embeds, cpu(get_pooled(pt_model, x_batch, nothing, modeltype)))
        end
    end
    train_input = hcat(train_embeds...)

    test_embeds = []
    for i in 1:batch_size:size(X_test, 2)
        b_idx = i:min(i+batch_size-1, size(X_test, 2))
        x_batch = gpu(Int32.(X_ranked[:, b_idx]))
        
        if use_pca
            x_pca_batch = gpu(Float32.(MultivariateStats.predict(pca_train_norm, Float32.(raw_test[:, b_idx]))))
            push!(test_embeds, cpu(get_pooled(pt_model, x_batch, x_pca_batch, modeltype)))
        else
            push!(test_embeds, cpu(get_pooled(pt_model, x_batch, nothing, modeltype)))
        end
    end
    test_input = hcat(test_embeds...)

    ft_model = Flux.Chain( # TODO: add another layer of size 512 if needed!
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.LayerNorm(hidden_dim),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications)
    ) |> gpu
end

opt = Flux.setup(Optimisers.Adam(lr), ft_model)

# train

train_losses = Float32[]
test_losses = Float32[]
preds = Int[]
trues = Int[]

train(ft_model, opt, train_input, y_train, test_input, y_test,
        epochs, train_losses, test_losses, preds, trues, ce_loss, 
        use_oversampling, cidx_dict, cs, false, batch_size
        )

acc = sum(preds .== trues) / length(trues)

# log stuff

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "finetuning", level, modeltype, timestamp)
mkpath(save_dir)

log_model(model, save_dir)
plot_loss(epochs, train_losses, test_losses, save_dir, "logit-ce", false, false)

jldsave(joinpath(save_dir, "losses.jld2"); 
        epochs = 1:epochs, 
        train_losses = train_losses, 
        test_losses = test_losses)
jldsave(joinpath(save_dir, "predstrues.jld2"); 
    all_preds = preds, 
    all_trues = trues)
jldsave(joinpath(save_dir, "indices.jld2"); 
    train = train_indices, 
    test = test_indices)

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

params_txt = joinpath(save_dir, "params.txt")
open(params_txt, "w") do io
    println(io, "PARAMETERS:")
    println(io, "###########")
    println(io, "gpu = $gpu_info")
    println(io, "pt1_epochs = $pt1_epochs")
    println(io, "pt2_epochs = $pt2_epochs")
    println(io, "dataset = $dataset")
    println(io, "batch_size = $batch_size")
    println(io, "ADDITIONAL NOTES: $additional_notes")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "accuracy = $acc")
end