# version #2 :)

using Pkg
if Sys.ARCH == :aarch64 || Sys.ARCH == :arm64
    Pkg.activate("/home/golem/scratch/chans/lincsv3/aarch64")
else
    Pkg.activate("/home/golem/scratch/chans/lincsv3")
end
Pkg.instantiate()

using DataFrames, Dates, StatsBase, JLD2
using LincsProject
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MultivariateStats, Zygote

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")


CUDA.device!(0)

# run-specific settings
# data_path = "data/lincs_untrt_data.jld2"
# dataset = "untrt"
n_epochs = 5
batch_size = 600 # 21 mins, 128 untrt
lr = lr * 6
gpu_info = CUDA.name(device())
additional_notes = "testing 5ep because idk why the 50ep crashed"

start_time = now()


### model structure



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
Flux.trainable(pe::PosEnc) = NamedTuple()

function (pe::PosEnc)(input::AbstractArray)
    seq_len = size(input,2)
    return input .+ @view(pe.pe_matrix[:, 1:seq_len]) # adds positional encoding to input embeddings
end


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


struct Model{E,J,P,D,T,C}
    embedding::E
    pca_proj::J
    # cls_token::L
    pos_encoder::P
    pos_dropout::D
    transformer::T
    classifier::C
end

function Model(;
    input_size::Int,
    pca_dim::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )
    embedding = Flux.Embedding(input_size => embed_dim)
    pca_proj = Flux.Dense(pca_dim => embed_dim)
    # cls_token = Flux.Embedding(1 => embed_dim)
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
    return Model(embedding, pca_proj, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@layer Model

function (model::Model)(input, input_pca) # input: n,b input_pca: p,b
    embedded = model.embedding(input) # e,n,b
    pca_embedded = model.pca_proj(input_pca) # e,b
    # cls_embedded = model.cls_token.weight # e,1

    # hybrid_emb = pca_embedded .+ cls_embedded # e,b
    # hybrid_reshaped = reshape(hybrid_emb, size(hybrid_emb, 1), 1, size(hybrid_emb, 2)) # e,1,b

    # pca_reshaped = reshape(pca_embedded, size(pca_embedded, 1), 1, size(pca_embedded, 2))
    # combined = cat(hybrid_reshaped, embedded, dims=2) # e,n+1,b

    pca_reshaped = reshape(pca_embedded, size(pca_embedded, 1), 1, size(pca_embedded, 2)) # e,1,b
    combined = embedded .+ pca_reshaped # e,n,b

    encoded = model.pos_encoder(combined) # e,n+1,b
    encoded_dropped = model.pos_dropout(encoded) # e,n+1,b

    transformed = model.transformer(encoded_dropped) # e,n+1,b

    # cls = transformed[:,1,:]
    # logits_output = model.classifier(cls)

    logits_output = model.classifier(transformed) # n,n+1,b
    
    return logits_output
end

# 130.452 ms (63236 allocations: 43.73 MiB) for the loss fxns below

function train_loss(model::Model, x, x_pca, y, n_classes)
    logits = model(x, x_pca)
    logits_flat = reshape(logits, size(logits, 1), :)
    y_flat = vec(y)

    # mask = Float32.((y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0))
    mask = (y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0)
    # mask = y_flat .!= Int32(-100)

    logits_masked = logits_flat[:, mask]
    y_masked = y_flat[mask]
    
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes)
    return Flux.logitcrossentropy(logits_masked, y_oh)
end

function test_loss(model::Model, x, x_pca, y, n_classes) 
    logits = model(x, x_pca)
    logits_flat = reshape(logits, size(logits, 1), :)
    y_flat = vec(y)

    mask = (y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0)
    # mask = y_flat .!= Int32(-100)

    logits_masked = logits_flat[:, mask]
    y_masked = y_flat[mask]

    y_oh = Flux.onehotbatch(y_masked, 1:n_classes)
    return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
end



### let's actually do it~



data = load(data_path)["filtered_data"]
data_expr = data.expr#[:, 1:10]
gene_medians = vec(median(data_expr, dims=2)) .+ 1e-10
X = rank_genes(data_expr, gene_medians)

n_features = size(X, 1) + 1
n_classes = size(X, 1)
n_genes = size(X, 1)
MASK_ID = (n_classes + 1)

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, -100, MASK_ID, false)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, -100, MASK_ID, false)

pca_train = fit(PCA, Float32.(X_train); maxoutdim=embed_dim)

model = Model(
    input_size=n_features,
    embed_dim=embed_dim,
    pca_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]

all_preds = Int[]
all_trues = Int[]
all_original_ranks = Int[]
all_prediction_errors = Int[]

for epoch in ProgressBar(1:n_epochs)
    epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))

        x_clean_cpu = X_train[:, start_idx:end_idx]
        x_pca_cpu = batch_pca(x_clean_cpu, pca_train, MASK_ID)

        x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
        x_pca = gpu(x_pca_cpu)
        y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
        
        loss_val, grads = Flux.withgradient(model) do m
            train_loss(m, x_gpu, x_pca, y_gpu, n_classes)
        end
        Flux.update!(opt, model, grads[1])
        # loss_val = loss(model, x_gpu, x_pca, y_gpu, "train")
        push!(epoch_losses, loss_val)
    end
    push!(train_losses, mean(epoch_losses))

    test_epoch_losses = Float32[]
    epoch_rank_errors = Int[]
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))

        x_clean_cpu = X_test[:, start_idx:end_idx]
        x_pca_cpu = batch_pca(x_clean_cpu, pca_train, MASK_ID)

        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        x_pca = gpu(x_pca_cpu)
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, logits_masked, y_masked = test_loss(model, x_gpu, x_pca, y_gpu, n_classes)
        push!(test_epoch_losses, test_loss_val)

        if isempty(y_masked) continue end

        logits_cpu = cpu(logits_masked)
        y_cpu = cpu(y_masked)
        
        if epoch == n_epochs
            y_cpu_batch = cpu(y_gpu)
            masked_indices_cartesian = findall(y_cpu_batch .!= -100)
            original_ranks_in_batch = [idx[1] for idx in masked_indices_cartesian]
        end

        for i in 1:length(y_cpu)
            true_gene_id = y_cpu[i]
            prediction_logits = logits_cpu[:, i]
            ranked_gene_ids = sortperm(prediction_logits, rev=true)
            predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids)
            
            if !isnothing(predicted_rank)
                error = predicted_rank - 1
                push!(epoch_rank_errors, error)
                
                if epoch == n_epochs
                    original_rank = original_ranks_in_batch[i] - 1
                    push!(all_original_ranks, original_rank)
                    push!(all_prediction_errors, error)
                end
            end
        end

        if epoch == n_epochs
            predicted_ids = Flux.onecold(logits_masked)
            append!(all_preds, cpu(predicted_ids))
            append!(all_trues, y_cpu)
        end
    end
    push!(test_losses, mean(test_epoch_losses))
    if !isempty(epoch_rank_errors)
        push!(test_rank_errors, mean(epoch_rank_errors))
    else
        push!(test_rank_errors, NaN32)
    end
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "rtf_v2", timestamp)
mkpath(save_dir)

plot_loss(n_epochs, train_losses, test_losses, save_dir, "logit-ce")
plot_rank_error(n_epochs, test_rank_errors, save_dir)
plot_boxplot(n_classes, all_trues, all_preds, save_dir)
plot_hexbin(all_trues, all_preds, "gene id", save_dir)
plot_prediction_error(all_original_ranks, all_prediction_errors, save_dir)

avg_errors = plot_mean_prediction_error(all_original_ranks, all_prediction_errors, save_dir)
cs, cp = plot_ranked_heatmap(all_trues, all_preds, save_dir, true)

log_model(model, save_dir)
embeddings = get_profile_embeddings(X, model)

# log run info
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

log_info(train_indices, test_indices, embeddings, n_epochs, 
                    train_losses, test_losses, all_preds, all_trues, 
                    all_original_ranks, all_prediction_errors, 
                    avg_errors, X_test_masked, y_test_masked, X_test)


log_tf_params(gpu_info, dataset, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)