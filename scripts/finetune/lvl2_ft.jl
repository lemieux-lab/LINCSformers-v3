# downstream task: prediction of perturbation (multi-class classification)
    # easier version would be treated vs. untreated profile via pert_type

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
#TODO: likely don't need aarch64 config here because its kinda deezed for this task?
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")
include("train.jl")

# settings

level = "lvl2"
modeltype = "rtf"
include("$(modeltype)_structs.jl")
if modeltype == "rtf"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-14_13-58"
elseif modeltype == "v1"
    dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-11_10-42"
elseif modeltype == "v2"
    dir = nothing #TODO: change after v2 finished running
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

pt1_epochs = 1
pt2_epochs = 1
additional_notes = "1ep test for rtf"

# if testing
# data_path = "data/lincs_untrt_data.jld2"
# dataset = "untrt

start_time = now()
CUDA.device!(0)

data = load(data_path)["filtered_data"]

X = data.expr 
y = data.inst.pert_id # pert_type = trt vs untrt, pert_id = actual perturbation type

#TODO: could probably also wrap all of this into a separate file 
#TODO: OR just use one file for everything and dictate the labels via data.inst.$idk defined in sbatch?
labels = unique(y)
n_classifications = length(labels)
ids = Dict(l => i for (i, l) in enumerate(labels))
y_ids = [ids[l] for l in y]
y_oh = Flux.onehotbatch(y_ids, 1:n_classifications)

n_genes = size(X, 1) 
n_classes_pt = n_genes
n_features_pt = n_classes_pt + 2 
CLS_ID = n_features_pt 

gene_medians = vec(median(X, dims=2)) .+ 1e-10
X_ranked = rank_genes(X, gene_medians)
CLS_VECTOR = fill(Int32(CLS_ID), (1, size(X_ranked, 2)))
X_input = vcat(CLS_VECTOR, X_ranked)

X_train, X_test, train_indices, test_indices = split_data(X_input, 0.2)
y_train = y_oh[:, train_indices]
y_test = y_oh[:, test_indices]

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
else
    error("modeltype not configured yet!")
end
Flux.loadmodel!(pt_model, state)

ft_model = FTModel(pt_model;
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    n_classifications=n_classifications
) |> gpu
opt = Flux.setup(Optimisers.Adam(lr), ft_model)

# pt1: gradient updates weights inside classifier not tf

Optimisers.freeze!(opt.pretrained) 

pt1_train_losses = Float32[]
pt1_test_losses = Float32[]
pt1_preds = Int[]
pt1_trues = Int[]

train(pt1_epochs, pt1_train_losses, pt1_test_losses, pt1_preds, pt1_trues, ce_loss)

# pt2: gradient updates both transformer and classifier weights

Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, lr/10) 

pt2_train_losses = Float32[]
pt2_test_losses = Float32[]
pt2_preds = Int[]
pt2_trues = Int[]

train(pt2_epochs, pt2_train_losses, pt2_test_losses, pt2_preds, pt2_trues, ce_loss)

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