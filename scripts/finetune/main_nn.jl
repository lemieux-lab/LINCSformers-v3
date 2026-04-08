# finetuning for mlp (full)
# nohup julia scripts/finetune/ft_mlp.jl > out/2026-04-03/ft_mlp.log 2>&1 &

# using Pkg
# Pkg.activate("/home/golem/scratch/chans/lincsv3")

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates, cuDNN, StatsBase

include("src/params.jl")
include("src/train.jl")
include("src/load_data.jl")
include("src/save.jl")

# settings
args_dict = Dict(replace(ARGS[i], "--" => "") => ARGS[i+1] for i in 1:2:(length(ARGS)-1))
for (key, val_str) in args_dict
    sym = Symbol(key)
    if hasproperty(config, sym)
        ExpectedType = fieldtype(typeof(config), sym)
        parsed_val = ExpectedType <: AbstractString ? val_str : parse(ExpectedType, val_str)
        setproperty!(config, sym, parsed_val)
    end
end

use_pca = false
use_oversmpl = config.level == "lvl2"

CUDA.device!(0)
gpu_info = CUDA.name(device())
if gpu_info == "Tesla V100-SXM2-32GB"
    config.batch_size = 128
else
    config.batch_size = 64
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", config.dataset, "finetuning", config.level, config.modeltype, "e2e", timestamp)
mkpath(save_dir)

# train
start_time = now()
data = load(config.data_path)["filtered_data"]

X_train, X_test, y_train, y_test, 
    train_indices, test_indices, 
    n_genes, n_classifications, 
    clsdict, cls, pca_info = dsplit(data, config)

model = Chain(
    Dense(n_genes => 512, gelu),
    LayerNorm(512),
    Dropout(config.drop_prob),
    Dense(512 => 256, gelu),
    LayerNorm(256),
    Dropout(config.drop_prob),
    Dense(256 => n_classifications)
) |> gpu

opt = Flux.setup(Optimisers.Adam(config.lr), model)

data_set = (X_train = X_train, ytrain = y_train, X_test = X_test, y_test = y_test, pca_train = nothing, pca_test = nothing)

config_set = (epochs = config.n_epochs, batch_size = config.batch_size, loss = ce_loss, 
              use_pca = false, use_oversmpl = use_oversmpl, clsdict = clsdict, cls = cls, 
              freq = config.cp_freq, save_dir = save_dir, pt = "mlp_e2e")

logs_set = (train_losses = Float32[], test_losses = Float32[], preds = Int[], trues = Int[])

train(model, opt, data_set, config_set, logs_set)

# log
run_time = now() - start_time
total_mins = div(run_time.value, 60000)
acc = sum(logs_set.preds .== logs_set.trues) / length(logs_set.trues)

save_run(save_dir, model, config.n_epochs, train_indices, test_indices, 
         logs_set.train_losses, logs_set.test_losses, logs_set.preds, logs_set.trues)

log_params(save_dir; gpu=gpu_info, epochs=config.n_epochs, dataset=config.dataset, 
           batch_size=config.batch_size, notes=config.additional_notes, 
           run_time="$(div(total_mins, 60))h $(rem(total_mins, 60))m", accuracy=acc)