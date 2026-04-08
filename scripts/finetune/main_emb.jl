# finetuning w/ same model, diff inputs

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
    else
        println("check ur key:'--$key'")
    end
end

use_pca = config.modeltype in ("v1", "v2")
use_oversmpl = config.level == "lvl2"
if haskey(args_dict, "dataset")
    config.data_path = "data/lincs_$(config.dataset)_data.jld2"
end
if config.modeltype != "mlp"
    include("../../structs/$(config.modeltype).jl")
end
if config.model_dir == ""
    if config.modeltype == "rtf"
        config.model_dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rank_tf/2026-03-24_02-55"
    elseif config.modeltype == "v1"
        config.model_dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-31_22-35"
    elseif config.modeltype == "v2"
        config.model_dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v2/2026-03-31_08-46"
    elseif config.modeltype != "mlp"
        error("check ur modeltype!!!")
    end
end

CUDA.device!(0)
gpu_info = CUDA.name(device())
if !haskey(args_dict, "batch_size")
    if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
        config.batch_size = 64
    elseif gpu_info == "Tesla V100-SXM2-32GB"
        config.batch_size = 128
    else
        config.batch_size = 64
    end
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", config.dataset, "finetuning", config.level, config.modeltype, "emb", timestamp)
mkpath(save_dir)

# train
start_time = now()
data = load(config.data_path)["filtered_data"]

data = load(config.data_path)["filtered_data"]

X_train, X_test, y_train, y_test, 
    train_indices, test_indices, n_genes, n_classifications, 
    clsdict, cls, pca_info = dsplit(data, config)

ft_model, train_input, test_input = emb(config, X_train, X_test, 
                                        pca_info, use_pca, n_genes, n_classifications)

opt = Flux.setup(Optimisers.Adam(config.lr), ft_model)


# pca is nothing bc the ft_model only takes the pooled embeddings
data_set = (X_train = train_input, ytrain = y_train, 
    X_test = test_input, y_test = y_test, 
    pca_train = nothing, pca_test = nothing)

config_set = (epochs = config.n_epochs, batch_size = config.batch_size, loss = ce_loss, 
    use_pca = false, use_oversmpl = use_oversmpl, 
    clsdict = clsdict, cls = cls, 
    freq = config.cp_freq, save_dir = save_dir, pt = "embed_ft")

logs_set = (train_losses = Float32[], test_losses = Float32[], 
    preds = Int[], trues = Int[])

train(ft_model, opt, data_set, config_set, logs_set)
acc = sum(logs_set.preds .== logs_set.trues) / length(logs_set.trues)

# log
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

save_run(save_dir, ft_model, config.n_epochs, train_indices, test_indices, 
         logs_set.train_losses, logs_set.test_losses, logs_set.preds, logs_set.trues)

log_params(save_dir; gpu=gpu_info, epochs=config.n_epochs, dataset=config.dataset, 
           batch_size=config.batch_size, notes=config.additional_notes, 
           run_time="$(run_hours)h $(run_minutes)m", accuracy=acc)