# main.jl

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
modeltype = "mlp"
pt1_epochs = 3
pt2_epochs = 12
additional_notes = "test"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversampling = level == "lvl2"

include("$(modeltype)_structs.jl")

if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-24_02-55"
elseif modeltype == "mlp"
    dir = nothing
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-11_10-42"
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
    X_train, X_test, train_indices, test_indices = split_data(X_ranked, 0.2)
    y_train = y_oh[:, train_indices]
    y_test  = y_oh[:, test_indices]
else
    pt_indices = load(joinpath(dir, "indices.jld2"))
    pt_train_indices = pt_indices["train_indices"]
    pt_test_indices  = pt_indices["test_indices"]

    idx_dict = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(idx))
    train_indices = [idx_dict[i] for i in pt_train_indices if haskey(idx_dict, i)]
    test_indices  = [idx_dict[i] for i in pt_test_indices if haskey(idx_dict, i)]

    X_train = X_ranked[:, train_indices]
    X_test  = X_ranked[:, test_indices]
    y_train = y_oh[:, train_indices]
    y_test  = y_oh[:, test_indices]

    if use_pca
        raw_train_pt = data.expr[:, pt_train_indices]
        pca_train_norm = fit(PCA, Float32.(raw_train_pt); maxoutdim=embed_dim)
        raw_train = X[:, train_indices]
        raw_test = X[:, test_indices]
    end
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
    ft_model = MLPModel(; 
        input_size = n_genes,
        hidden_dim = hidden_dim,
        n_classifications = n_classifications,
        drop_prob = drop_prob
    ) |> gpu
    
    opt = Flux.setup(Optimisers.Adam(lr), ft_model)
    
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

    ft_model = FTModel(pt_model;
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_classifications=n_classifications
    ) |> gpu
    
    opt = Flux.setup(Optimisers.Adam(lr), ft_model)
end


# pt1: gradient updates weights inside classifier not tf

Optimisers.freeze!(opt.pretrained) 

pt1_train_losses = Float32[]
pt1_test_losses = Float32[]
pt1_preds = Int[]
pt1_trues = Int[]

train(pt1_epochs, pt1_train_losses, pt1_test_losses, pt1_preds, pt1_trues, ce_loss, use_oversampling, cidx_dict, cs, use_pca, batch_size)

# pt2: gradient updates both transformer and classifier weights

Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, lr/10) 

pt2_train_losses = Float32[]
pt2_test_losses = Float32[]
pt2_preds = Int[]
pt2_trues = Int[]

train(pt2_epochs, pt2_train_losses, pt2_test_losses, pt2_preds, pt2_trues, ce_loss, use_oversampling, cidx_dict, cs, use_pca, batch_size)

# log stuff

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "finetuning", level, modeltype, timestamp)
mkpath(save_dir)

log_model(ft_model, save_dir)
plot_loss(pt1_epochs, pt1_train_losses, pt1_test_losses, save_dir, "logit-ce", true, false)
plot_loss(pt2_epochs, pt2_train_losses, pt2_test_losses, save_dir, "logit-ce", false, true)

jldsave(joinpath(save_dir, "pt1_losses.jld2"); 
        epochs = 1:pt1_epochs, 
        train_losses = pt1_train_losses, 
        test_losses = pt1_test_losses)
jldsave(joinpath(save_dir, "pt2_losses.jld2"); 
        epochs = 1:pt2_epochs, 
        train_losses = pt2_train_losses, 
        test_losses = pt2_test_losses)
jldsave(joinpath(save_dir, "pt1_predstrues.jld2"); 
    all_preds = pt1_preds, 
    all_trues = pt1_trues)
jldsave(joinpath(save_dir, "pt2_predstrues.jld2"); 
    all_preds = pt2_preds, 
    all_trues = pt2_trues)
jldsave(joinpath(save_dir, "indices"); 
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
end