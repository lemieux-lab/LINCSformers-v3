# using Pkg
# Pkg.activate("/home/golem/scratch/chans/lincsv3")

using DataFrames, Dates, StatsBase, JLD2, LincsProject
using Flux, Random, ProgressBars, CUDA, cuDNN, Statistics, CairoMakie, LinearAlgebra, MultivariateStats

include("src/params.jl")
include("src/structs.jl")
include("src/fxns.jl")
include("src/plot.jl")
include("src/save.jl")

# run-specific settings
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
if haskey(args_dict, "modeltype")
    mode_map = Dict("rtf" => :none, "v1" => :concat, "v2" => :add)
    config.pca_mode = mode_map[config.modeltype]
end
if haskey(args_dict, "dataset")
    config.data_path = "data/lincs_$(config.dataset)_data.jld2" # fyi trt refers to trt and untrt, untrt is untrt only
end
gpu_info = CUDA.name(device())
if !haskey(args_dict, "batch_size")
    if gpu_info == "NVIDIA GeForce GTX 1080 Ti"
        config.batch_size = 42
    elseif gpu_info == "Tesla V100-SXM2-32GB"
        config.batch_size = 128
    elseif gpu_info == "NVIDIA GH200 144G HBM3e"
        config.batch_size = 600
        config.lr *= 6
    else
        config.batch_size = 64
    end
end

# start here
CUDA.device!(0)
start_time = now()

data = load(config.data_path)["filtered_data"]
data_expr = data.expr[:, 1:1000]
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

pca_train_norm = fit(PCA, Float32.(raw_train); maxoutdim=config.embed_dim)
if config.pca_mode != :none # this is the same thing as using predict(), y = P'(X-mu)
    pgpu = gpu(MultivariateStats.projection(pca_train_norm)')
    mgpu = gpu(MultivariateStats.mean(pca_train_norm))
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
    pca_mode=config.pca_mode) |> gpu

opt = Flux.setup(Adam(config.lr), model)

train_losses, test_losses, test_rank_errors = Float32[], Float32[], Float32[]
final_preds, final_trues = Int[], Int[]

X_train_masked_gpu = gpu(Int.(X_train_masked))
y_train_masked_gpu = gpu(Int.(y_train_masked))
X_test_masked_gpu = gpu(Int.(X_test_masked))
y_test_masked_gpu = gpu(Int.(y_test_masked))
raw_train_gpu = gpu(Float32.(raw_train))
raw_test_gpu = gpu(Float32.(raw_test))

for epoch in ProgressBar(1:config.n_epochs)
    train_loss, _, _, _, _, _ = train_epoch(model, opt, X_train_masked_gpu, y_train_masked_gpu, raw_train_gpu, 
                                            pgpu, mgpu, MASK_ID, n_classes, config.batch_size; 
                                            mode=:train, is_final_epoch=false)
    push!(train_losses, train_loss)
    
    is_final = (epoch == config.n_epochs)
    test_loss, test_err, preds, trues, orig_ranks, pred_errs = train_epoch(model, opt, X_test_masked_gpu, y_test_masked_gpu, raw_test_gpu, 
                                                                            pgpu, mgpu, MASK_ID, n_classes, config.batch_size; 
                                                                            mode=:test, is_final_epoch=is_final)
    push!(test_losses, test_loss)
    push!(test_rank_errors, test_err)
    if is_final
        final_preds, final_trues = preds, trues
        final_original_ranks, final_prediction_errors = orig_ranks, pred_errs
    end
end

# log
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", config.dataset, config.modeltype, timestamp)
mkpath(save_dir)

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