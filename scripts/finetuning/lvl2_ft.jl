# downstream task: treated vs. untreated profile
    # can do prediction of perturbation type if we want multi-class classification instead

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
# Pkg.Registry.add(RegistrySpec(url="git@github.com:lemieux-lab/LabRegistry.git"))
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")
include("ft_structs.jl")

start_time = now()
CUDA.device!(0)

# run-specific settings
data_path = "data/lincs_untrt_data.jld2"
dataset = "trt"
gpu_info = CUDA.name(device())
additional_notes = "ft 1ep test on multiclass (pert_id)"
batch_size = 128

# okkkkk lets go! 
# TODO: makes more sense to wrap pt1 and pt2 train loops into functions in 
        # separate files and import them for different downstream tasks 

data = load(data_path)["filtered_data"]

X = data.expr 
y = data.inst.pert_id # pert_type = trt vs untrt, pert_id = actual perturbation type

#=
for reference:
    
julia> data.compound.
canonical_smiles
first_name
inchi_key
pert_id

julia> data.inst.
bead_batch      build_name      cell_iname      cell_mfc_name   cmap_name       count_cv
count_mean      det_plate       det_well        dyn_range       failure_mode    inv_level_10
nearest_dose    pert_dose       pert_dose_unit  pert_id         pert_idose      pert_itime
pert_mfc_id     pert_time       pert_time_unit  pert_type       project_code    qc_f_logp
qc_iqr          qc_pass         qc_slope        rna_plate       rna_well        sample_id

julia> data.gene.
ensembl_id     feature_space
gene_id        gene_symbol
gene_title     gene_type
=#

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

dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54"
state = load("$dir/model_state.jld2")["model_state"]
pt_model = Model(
    input_size=n_features_pt,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes_pt,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
)
Flux.loadmodel!(pt_model, state)

ft_model = FTModel(pt_model;
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    n_classifications=n_classifications
) |> gpu
opt = Flux.setup(Optimisers.Adam(lr), ft_model)
Optimisers.freeze!(opt.pretrained) 


# classification
function loss(model, x, y)
    logits = model(x)
    return Flux.logitcrossentropy(logits, y)
end

#= 
train via:

pass input w/ labels to model; 
classifier takes CLS token to make prediction; 
check against y labels; 
gradient updates weights inside classifier not tf;
=#

pt1_epochs = 2
pt1_train_losses = Float32[]
pt1_test_losses = Float32[]
pt1_preds = Int[]
pt1_trues = Int[]

for epoch in ProgressBar(1:pt1_epochs)
    train_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))

        x_gpu = gpu(Int32.(X_train[:, start_idx:end_idx]))
        y_gpu = gpu(Float32.(y_train[:, start_idx:end_idx]))

        lv, grads = Flux.withgradient(ft_model) do m
            loss(m, x_gpu, y_gpu)
        end
        Flux.update!(opt, ft_model, grads[1])
        train_loss_val = loss(ft_model, x_gpu, y_gpu)
        push!(train_epoch_losses, train_loss_val)
    end
    push!(pt1_train_losses, mean(train_epoch_losses))

    test_epoch_losses = Float32[]

    for start_idx in 1:batch_size:size(X_test, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test, 2))

        x_gpu = gpu(Int32.(X_test[:, start_idx:end_idx]))
        y_gpu = gpu(Float32.(y_test[:, start_idx:end_idx]))

        test_loss_val = loss(ft_model, x_gpu, y_gpu)
        push!(test_epoch_losses, test_loss_val)

        logits = ft_model(x_gpu)
        test_loss_val = Flux.logitcrossentropy(logits, y_gpu)

        if epoch == pt1_epochs
            preds = Flux.onecold(cpu(logits))
            trues = Flux.onecold(cpu(y_gpu))
            append!(pt1_preds, preds)
            append!(pt1_trues, trues)
        end
    end
    push!(pt1_test_losses, mean(test_epoch_losses))
end


Optimisers.thaw!(opt.pretrained)
Optimisers.adjust!(opt, lr/10) 

#=
train again via:

reduce learning rate
gradient updates both transformer and classifier weights;
=#


pt2_epochs = 6
pt2_train_losses = Float32[]
pt2_test_losses = Float32[]
pt2_preds = Int[]
pt2_trues = Int[]

for epoch in ProgressBar(1:pt2_epochs)
    train_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))

        x_gpu = gpu(Int32.(X_train[:, start_idx:end_idx]))
        y_gpu = gpu(Float32.(y_train[:, start_idx:end_idx]))

        lv, grads = Flux.withgradient(ft_model) do m
            loss(m, x_gpu, y_gpu)
        end
        Flux.update!(opt, ft_model, grads[1])
        train_loss_val = loss(ft_model, x_gpu, y_gpu)
        push!(train_epoch_losses, train_loss_val)
    end
    push!(pt2_train_losses, mean(train_epoch_losses))

    test_epoch_losses = Float32[]

    for start_idx in 1:batch_size:size(X_test, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test, 2))

        x_gpu = gpu(Int32.(X_test[:, start_idx:end_idx]))
        y_gpu = gpu(Float32.(y_test[:, start_idx:end_idx]))

        test_loss_val = loss(ft_model, x_gpu, y_gpu)
        push!(test_epoch_losses, test_loss_val)

        logits = ft_model(x_gpu)
        test_loss_val = Flux.logitcrossentropy(logits, y_gpu)

        if epoch == pt2_epochs
            preds = Flux.onecold(cpu(logits))
            trues = Flux.onecold(cpu(y_gpu))
            append!(pt2_preds, preds)
            append!(pt2_trues, trues)
        end
    end
    push!(pt2_test_losses, mean(test_epoch_losses))
end


# log stuff

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "finetune", timestamp)
mkpath(save_dir)

log_model(ft_model, save_dir)
plot_loss(pt1_epochs, pt1_train_losses, pt1_test_losses, save_dir, "logit-ce", true, false)
plot_loss(pt2_epochs, pt2_train_losses, pt2_test_losses, save_dir, "logit-ce", false, true)

jldsave(joinpath(save_dir, "pt1_losses.jld2"); 
        epochs = 1:pt1_epochs, 
        train_losses = pt1_train_losses, 
        test_losses = pt1_test_losses
)
jldsave(joinpath(save_dir, "pt2_losses.jld2"); 
        epochs = 1:pt2_epochs, 
        train_losses = pt2_train_losses, 
        test_losses = pt2_test_losses
)
jldsave(joinpath(save_dir, "pt1_predstrues.jld2"); 
    all_preds = pt1_preds, 
    all_trues = pt1_trues
)
jldsave(joinpath(save_dir, "pt2_predstrues.jld2"); 
    all_preds = pt2_preds, 
    all_trues = pt2_trues
)

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