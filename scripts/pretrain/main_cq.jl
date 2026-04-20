# using Pkg
# Pkg.activate("/home/golem/scratch/chans/lincsv4")
# TODO: do aarch pkg in sbatch file, make sure ArgParse is in the pkg manager for this via interactive run on balrog first
using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv4/cq") # if issues then restart julia language server!
using Dates, StatsBase, JLD2
# using cuDNN
using CUDA
# using LuxCUDA
using Functors
using Flux, Random, ProgressBars, Statistics, CairoMakie, LinearAlgebra, MultivariateStats

include("src/params.jl")
include("src/structs.jl")
include("src/fxns.jl")
include("src/plot.jl")
include("src/save.jl")

# run-specific settings
args = load_args()
kwargs = Dict(Symbol(k) => v for (k, v) in args)
config = Config(; kwargs...)

config = Config()

mode_map = Dict("rtf" => :none, "v1" => :concat, "v2" => :add)
config.pca_mode = mode_map[config.modeltype]

# config.data_path = "data/lincs_trt_data.jld2"

# if haskey(kwargs, "dataset")
#     config.data_path = "data/lincs_$(config.dataset)_data.jld2" # fyi trt refers to trt and untrt, untrt is untrt only
# end

# config.batch_size = 64

gpu_info = CUDA.name(device())
if !haskey(kwargs, "batch_size")
    if gpu_info ∈ ("NVIDIA GeForce GTX 1080 Ti", "NVIDIA GeForce RTX 4090")
        config.batch_size = 42
    elseif gpu_info == "NVIDIA GH200 144G HBM3e MIG 1g.18gb"
        config.batch_size = 64
    elseif gpu_info ∈ ("Tesla V100-SXM2-32GB", "NVIDIA RTX A4500", "NVIDIA RTX 6000 Ada Generation")
        config.batch_size = 128
    elseif gpu_info == "NVIDIA GH200 144G HBM3e"
        config.batch_size = 500
        config.lr *= 3
    else
        config.batch_size = 64
    end
end

# wandb settings
# include("deps/build.jl")
using PyCall
println("using pycall from: ", PyCall.python)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
run_name = "$(timestamp)_pretrain_$(config.modeltype)"
wandb = pyimport("wandb")
config_dict = Dict(String(k) => getfield(config, k) for k in fieldnames(typeof(config)))
wandb.init(project="LINCSformers", config=config_dict, mode=config.wandb_mode, name=run_name)


# start here
CUDA.device!(0)
start_time = now()

data = load("/home/chans/links/scratch/lincsv4/data/data_expr.jld2")["data_expr"]
data_expr = data#[:, 1:1000]
gene_medians = vec(median(data_expr, dims=2)) .+ 1e-10
X = rank_genes(data_expr, gene_medians)

n_features = size(X, 1) + 1
n_classes = size(X, 1)
MASK_ID = n_classes + 1

X_train, X_test, train_indices, test_indices = split_data(X, 0.2);
X_train_masked, y_train_masked = mask_input(X_train, config.mask_ratio, -100, MASK_ID, false);
X_test_masked, y_test_masked = mask_input(X_test, config.mask_ratio, -100, MASK_ID, false);

raw_train = data_expr[:, train_indices]
raw_test = data_expr[:, test_indices]

pca_train_norm = fit(PCA, Float32.(raw_train); maxoutdim=config.embed_dim);
if config.pca_mode != :none # this is the same thing as using predict(), y = P'(X-mu)
    pgpu = gpu(MultivariateStats.projection(pca_train_norm)');
    mgpu = gpu(MultivariateStats.mean(pca_train_norm));
else
    pgpu = nothing
    mgpu = nothing
end

model = Model(
    input_size=n_features, 
    pca_dim=config.embed_dim, 
    embed_dim=config.embed_dim,
    n_layers=config.n_layers, 
    n_classes=n_classes, 
    n_heads=config.n_heads,
    hidden_dim=config.hidden_dim, 
    dropout_prob=config.drop_prob,
    pca_mode=config.pca_mode)
    
opt = Flux.setup(Adam(config.lr), model)

model = fmap(manual_gpu_transfer, model; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x))
opt = fmap(manual_gpu_transfer, opt; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x))

base_save_dir = joinpath("plots", config.dataset, config.modeltype)
latest_run_dir, latest_cp_file, latest_epoch = find_latest_checkpoint(base_save_dir)

start_epoch = 1

if latest_cp_file !== nothing && latest_epoch < config.n_epochs
    println("resuming from checkpoint: $latest_cp_file")
    
    save_dir = latest_run_dir 
    start_epoch = latest_epoch + 1

    cp_data = load(latest_cp_file)
    Flux.loadmodel!(model, cp_data["model_state"]) 
    
    model = fmap(manual_gpu_transfer, model; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x))
    opt = fmap(manual_gpu_transfer, cp_data["opt_state"]; exclude = x -> x isa Flux.Dropout || Functors.isleaf(x))
        
    train_losses = cp_data["train_losses"]
    test_losses = cp_data["test_losses"]
    test_rank_errors = cp_data["test_rank_errors"] 
    
    final_preds = cp_data["checkpt_preds"]
    final_trues = cp_data["checkpt_trues"]

else
    println("starting new run...")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    save_dir = joinpath(base_save_dir, timestamp)
    mkpath(save_dir)
    
    train_losses, test_losses, test_rank_errors = Float32[], Float32[], Float32[]
    final_preds, final_trues = Int[], Int[]
end

# X_train_masked_gpu = gpu(Int.(X_train_masked))
# y_train_masked_gpu = gpu(Int.(y_train_masked))
# X_test_masked_gpu = gpu(Int.(X_test_masked))
# y_test_masked_gpu = gpu(Int.(y_test_masked))
# raw_train_gpu = gpu(Float32.(raw_train))
# raw_test_gpu = gpu(Float32.(raw_test))

final_original_ranks, final_prediction_errors = Int[], Int[]
freq = config.freq
warmup_epochs = max(1, div(config.n_epochs, 10))
for epoch in ProgressBar(start_epoch:config.n_epochs)
    Optimisers.adjust!(opt, compute_lr(epoch, config.n_epochs, config.lr, warmup_epochs))
    train_loss, _, _, _, _, _ = train_epoch(model, opt, X_train_masked, y_train_masked, raw_train,
                                            pgpu, mgpu, MASK_ID, n_classes, config.batch_size; 
                                            mode=:train, is_final_epoch=false)
    push!(train_losses, train_loss)
    
    is_final = (epoch == config.n_epochs)
    test_loss, test_err, preds, trues, orig_ranks, pred_errs = train_epoch(model, opt, X_test_masked, y_test_masked, raw_test, 
                                                                            pgpu, mgpu, MASK_ID, n_classes, config.batch_size; 
                                                                            mode=:test, is_final_epoch=is_final)
    push!(test_losses, test_loss)
    push!(test_rank_errors, test_err)

    if config.wandb_mode != "disabled"
        wandb.log(Dict(
            "epoch" => epoch,
            "train_loss" => train_loss,
            "test_loss" => test_loss,
            "test_rank_error" => test_err 
        ))
    end

    if is_final
        append!(empty!(final_preds), preds)
        append!(empty!(final_trues), trues)
        append!(empty!(final_original_ranks), orig_ranks)
        append!(empty!(final_prediction_errors), pred_errs)
    end

    is_checkpt = (!isnothing(freq) && epoch % freq == 0) || is_final
    if is_checkpt
        cp_dir = joinpath(save_dir, "checkpts")
        mkpath(cp_dir) 
        path = joinpath(cp_dir, "epoch_$(epoch).jld2")
        jldsave(path; 
            model_state = Flux.state(cpu(model)),
            model_object = cpu(model),
            opt_state = cpu(opt), 
            train_losses = train_losses, 
            test_losses = test_losses, 
            test_rank_errors = test_rank_errors,
            checkpt_preds = preds, 
            checkpt_trues = trues)
    end
end

# log
plot_loss(config.n_epochs, train_losses, test_losses, save_dir, "logit-ce")
plot_rank_error(config.n_epochs, test_rank_errors, save_dir)
plot_hexbin(final_trues, final_preds, "gene id", save_dir)
plot_prediction_error(final_original_ranks, final_prediction_errors, save_dir)
# avg_errors = plot_mean_prediction_error(final_original_ranks, final_prediction_errors, save_dir)
cs, cp = plot_ranked_heatmap(final_trues, final_preds, save_dir, true)

log_model(model, save_dir)
# embeddings = get_profile_embeddings(X, data_expr, model, pgpu, mgpu, config.batch_size)

run_time = now() - start_time
total_minutes = div(run_time.value, 60000)

log_info(train_indices, test_indices, 
            nothing, config.n_epochs, 
            train_losses, test_losses, final_preds, 
            final_trues, X_test_masked, y_test_masked, X_test, 
            save_dir)
log_tf_params(config, gpu_info, div(total_minutes, 60), rem(total_minutes, 60), save_dir)

wandb.finish()