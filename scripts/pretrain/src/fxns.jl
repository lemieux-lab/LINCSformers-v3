using Flux, CUDA, StatsBase, Statistics, Random, ArgParse

function load_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
            default = 1
            required = true
        "--modeltype", "-t"
            help = "model type: rtf, v1, or v2"
            arg_type = String
            required = true
        "--dataset", "-d"
            help = "dataset: trt or untrt"
            arg_type = String
            default = "trt"
            required = false
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
            default = 64
            required = false
        "--note", "-n"
            help = "run-specific notes"
            arg_type = String
            required = false
    end
    return parse_args(s)
end

function rank_genes(expr, medians)
    n, m = size(expr)
    data_ranked = Matrix{Int32}(undef, size(expr)) 
    normalized_col = Vector{Float32}(undef, n) 
    sorted_ind_col = Vector{Int32}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        sortperm!(sorted_ind_col, normalized_col, rev=true)
        for i in 1:n
            data_ranked[i, j] = sorted_ind_col[i]
        end
    end
    return data_ranked
end

function split_data(X, ratio::Float64, y=nothing)
    n = size(X, 2)
    idx = shuffle(1:n)
    n_test = floor(Int, n*ratio)
    test_idx, train_idx = idx[1:n_test], idx[n_test+1:end]
    if isnothing(y) 
        return X[:, train_idx], X[:, test_idx], train_idx, test_idx 
    end
    return X[:, train_idx], y[:, train_idx], X[:, test_idx], y[:, test_idx], train_idx, test_idx
end

function mask_input(X::Matrix, mask_ratio::Float64, mask_val, mask_id, offset::Bool=false)
    X_masked = copy(X)
    mask_labels = fill!(similar(X), mask_val)
    n_rows, n_samples = size(X)
    idx = (1 + offset):n_rows
    num_masked = ceil(Int, length(idx) * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(idx, num_masked, replace=false) 
        for pos in mask_pos
            mask_labels[pos, j] = X[pos, j]
            X_masked[pos, j] = mask_id 
        end
    end
    return X_masked, mask_labels
end

function masked_logitcrossentropy(logits, y, n_classes)
    logits_flat = reshape(logits, size(logits, 1), :)
    y_flat = vec(y)
    mask = (y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0)
    if !any(mask) 
        return 0.0f0, nothing, nothing 
    end
    
    y_masked = y_flat[mask]
    logits_masked = logits_flat[:, mask]
    
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes)
    return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
end

function train_epoch(model, opt, X_masked, y_masked, raw_data, 
                    pca_gpu, mean_gpu, MASK_ID, n_classes, batch_size; 
                    mode=:train, is_final_epoch::Bool=false)
    epoch_losses = Float32[]
    all_preds, all_trues, epoch_rank_errors = Int[], Int[], Int[]
    all_original_ranks, all_prediction_errors = Int[], Int[] 
    mode == :train ? Flux.trainmode!(model) : Flux.testmode!(model)
    
    for start_idx in 1:batch_size:size(X_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_masked, 2))

        x_batch = CuArray(X_masked[:, start_idx:end_idx]) 
        y_batch = CuArray(y_masked[:, start_idx:end_idx])
        
        if pca_gpu !== nothing
            raw_batch = CuArray(raw_data[:, start_idx:end_idx])
            x_raw_masked = (x_batch .!= MASK_ID) .* raw_batch
            x_pca = pca_gpu * (x_raw_masked .- mean_gpu)
        else
            x_pca = nothing
        end
        
        if mode == :train
            l_val, grads = Flux.withgradient(model) do m
                l, _, _ = masked_logitcrossentropy(m(x_batch, x_pca), y_batch, n_classes)
                return l
        end
            Flux.update!(opt, model, grads[1])
            l_val = masked_logitcrossentropy(model(x_batch, x_pca), y_batch, n_classes)
            push!(epoch_losses, l_val)
        else
            loss_val, logits_masked, y_targets = masked_logitcrossentropy(model(x_batch, x_pca), y_batch, n_classes)
            push!(epoch_losses, loss_val)
            if !isempty(y_targets)
                logits_cpu = cpu(logits_masked)
                y_targets_cpu = cpu(y_targets)
                append!(all_trues, y_targets_cpu)
                append!(all_preds, Flux.onecold(logits_cpu))
                if is_final_epoch 
                    y_cpu_batch = cpu(y_batch)
                    midx = findall(y -> y != -100 && 0 < y <= n_classes, y_cpu_batch)
                    original_ranks_in_batch = [idx[1] for idx in midx]
                end
                for i in 1:length(y_targets_cpu) 
                    target_id = y_targets_cpu[i]
                    target_logit = logits_cpu[target_id, i]
                    error = count(x -> x > target_logit, @view(logits_cpu[:, i]))
                    push!(epoch_rank_errors, error)
                    if is_final_epoch
                        push!(all_original_ranks, original_ranks_in_batch[i] - 1)
                        push!(all_prediction_errors, error)
                    end
                end
            end
        end
    end
    return mean(epoch_losses), mean(epoch_rank_errors), all_preds, all_trues, all_original_ranks, all_prediction_errors
end