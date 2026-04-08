# finetuning for transformers (full)

using Pkg # TODO: put this in the slurm .sh file instead
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

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
include("../../structs/$(config.modeltype).jl")
if config.model_dir == ""
    dirs = Dict("rtf"=>"rank_tf/2026-03-24_02-55", "v1"=>"rtf_v1/2026-03-31_22-35", "v2"=>"rtf_v2/2026-03-31_08-46")
    config.model_dir = "/home/golem/scratch/chans/lincsv3/plots/trt/$(dirs[config.modeltype])"
end

CUDA.device!(0)
gpu_info = CUDA.name(device())
if gpu_info == "Tesla V100-SXM2-32GB"
    config.batch_size = 128
else
    config.batch_size = 64
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", config.dataset, "finetuning", config.level, config.modeltype, timestamp)
mkpath(save_dir)

# train
start_time = now()

data = load(config.data_path)["filtered_data"]
X_train, X_test, y_train, y_test, train_indices, test_indices, n_genes, n_classifications, clsdict, cls, pcainfo = dsplit(data, config)

model, X_pca_train, X_pca_test = mstate(config, pcainfo, use_pca, n_classifications)
opt = Flux.setup(Optimisers.Adam(config.lr), model)

data_set = (X_train=X_train, ytrain=y_train, X_test=X_test, y_test=y_test, pca_train=X_pca_train, pca_test=X_pca_test)

pt1_epochs = config.pt1_epochs
pt2_epochs = config.pt2_epochs

# pt1: gradient updates weights inside classifier not tf
logs_pt1 = (train_losses=Float32[], test_losses=Float32[], preds=Int[], trues=Int[])
train(model, opt, data_set, (epochs=pt1_epochs, batch_size=config.batch_size, loss=ce_loss, use_pca=use_pca, use_oversmpl=use_oversmpl, clsdict=clsdict, cls=cls, freq=config.cp_freq, save_dir=save_dir, pt="pt1"), logs_pt1)

acc_pt1 = sum(logs_pt1.preds .== logs_pt1.trues) / length(logs_pt1.trues)
save_run(save_dir, model, pt1_epochs, train_indices, test_indices, logs_pt1.train_losses, logs_pt1.test_losses, logs_pt1.preds, logs_pt1.trues, prefix="pt1_")

# pt2: gradient updates both transformer and classifier weights
Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, config.lr / 10) 

logs_pt2 = (train_losses=Float32[], test_losses=Float32[], preds=Int[], trues=Int[])
train(model, opt, data_set, (epochs=pt2_epochs, batch_size=config.batch_size, loss=ce_loss, use_pca=use_pca, use_oversmpl=use_oversmpl, clsdict=clsdict, cls=cls, freq=config.cp_freq, save_dir=save_dir, pt="pt2"), logs_pt2)

acc_pt2 = sum(logs_pt2.preds .== logs_pt2.trues) / length(logs_pt2.trues)
save_run(save_dir, model, pt2_epochs, train_indices, test_indices, logs_pt2.train_losses, logs_pt2.test_losses, logs_pt2.preds, logs_pt2.trues, prefix="pt2_")

# log
run_time = now() - start_time
total_minutes = div(run_time.value, 60000)

log_params(save_dir; gpu=gpu_info, pt1_epochs=pt1_epochs, pt2_epochs=pt2_epochs, dataset=config.dataset, 
            batch_size=config.batch_size, notes=config.additional_notes, 
            run_time="$(div(total_minutes, 60))h $(rem(total_minutes, 60))m", 
            pt1_accuracy=acc_pt1, pt2_accuracy=acc_pt2)