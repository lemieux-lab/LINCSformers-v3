# finetuning for mlp (full)

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
epochs = 100
additional_notes = "even longer test w/o lr decrease"

# setup

use_pca = modeltype in ("v1", "v2")
use_oversampling = level == "lvl2"

if modeltype != "mlp"
    error("wrong file >:(")
end

gpu_info = CUDA.name(device())
if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
    batch_size = 64
elseif gpu_info == "Tesla V100-SXM2-32GB"
    batch_size = 128
else
    batch_size = 64
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

# gene_medians = vec(median(X, dims=2)) .+ 1e-10 # TODO: not sure if this is needed??? probably not

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
y_train = y_oh[:, train_indices]
y_test = y_oh[:, test_indices]

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

model = Chain(
    Dense(n_genes => 512),
    LayerNorm(512),
    Dropout(drop_prob),
    Dense(512 => 256),
    LayerNorm(256),
    Dropout(drop_prob),
    Dense(128 => n_classifications)
    ) |> gpu

opt = Flux.setup(Optimisers.Adam(lr), model)

train_losses = Float32[]
test_losses = Float32[]
preds = Int[]
trues = Int[]

train(model, epochs, train_losses, test_losses, preds, trues, ce_loss, use_oversampling, cidx_dict, cs, use_pca, batch_size)

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
    println(io, "epochs = $epochs")
    println(io, "dataset = $dataset")
    println(io, "batch_size = $batch_size")
    println(io, "ADDITIONAL NOTES: $additional_notes")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "accuracy = $acc")
end