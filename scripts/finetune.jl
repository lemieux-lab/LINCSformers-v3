using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.Registry.add(RegistrySpec(url="git@github.com:lemieux-lab/LabRegistry.git"))
Pkg.instantiate()

# can use /home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54; this is 10ep run on smaug
# nohup julia scripts/finetune.jl > ft_untrt_check_feb26.log 2>&1 &

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates

include("../src/params.jl")
include("../src/fxns.jl")
include("../src/plot.jl")
include("../src/save.jl")

# data_path = "data/lincs_untrt_data.jld2"
# dataset = "untrt"
# batch_size = 100

start_time = now()
CUDA.device!(0)

# pretrained model struct for reconstruction


### positional encoder
struct PosEnc{A<:AbstractArray}
    pe_matrix::A
end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(pe_matrix)
end

Flux.@layer PosEnc
Flux.trainable(pe::PosEnc) = ()

function (pe::PosEnc)(input::AbstractArray)
    seq_len = size(input,2)
    return input .+ @view(pe.pe_matrix[:, 1:seq_len]) # adds positional encoding to input embeddings
end

### transformer
struct Transf{A,D,N,M}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M 
    mlp_norm::N
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@layer Transf

function (tf::Transf)(input) # input shape: embed_dim × seq_len × batch_size
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

### full model as << ranked data --> token embedding --> position embedding --> transformer --> classifier head >>
struct Model{E,P,D,T,C}
    embedding::E
    pos_encoder::P
    pos_dropout::D
    transformer::T
    classifier::C
end

function Model(;
    input_size::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    embedding = Flux.Embedding(input_size => embed_dim)
    pos_encoder = PosEnc(embed_dim, input_size)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )
    return Model(embedding, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@layer Model

function (model::Model)(input)
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    logits_output = model.classifier(transformed)
    return logits_output
end


# finetuned struct to stick them together


struct FTModel{P,H}
    pretrained::P
    head::H
end

Flux.@layer FTModel

function FTModel(pt_model;
    embed_dim::Int,
    hidden_dim::Int,
    n_classifications::Int,
    )
    pretrained = (
        embedding = pt_model.embedding,
        pos_encoder = pt_model.pos_encoder,
        pos_dropout = pt_model.pos_dropout,
        transformer = pt_model.transformer
    )
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications)
        )
    return FTModel(pretrained, head)
end

function (m::FTModel)(input)
    embedded = m.pretrained.embedding(input)
    encoded = m.pretrained.pos_encoder(embedded)
    encoded_dropped = m.pretrained.pos_dropout(encoded)
    transformed = m.pretrained.transformer(encoded_dropped)
    cls_token = transformed[:, 1, :]
    return m.head(cls_token)
end


# okkkkk lets go!

# say i guess we want to do the easiest; cell line identification

data = load(data_path)["filtered_data"]

X = data.expr 
y = data.inst.cell_mfc_name # or is it cell_iname?

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

# need to process X and y

cell_lines = unique(y)
n_classifications = length(cell_lines)
cell_ids = Dict(cell => i for (i, cell) in enumerate(cell_lines))
y_ids = [cell_ids[cell] for cell in y]
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

pt1_epochs = 5
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


pt2_epochs = 20
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