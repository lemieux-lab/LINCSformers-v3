using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")

using Flux, CUDA, Random, OneHotArrays

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

function split_data(X, test_ratio::Float64, y=nothing) # masking doesn't need y!
    n = size(X, 2)
    indices = shuffle(1:n)
    test_size = floor(Int, n * test_ratio)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    X_train = X[:, train_indices]
    X_test = X[:, test_indices]
    if y === nothing
        return X_train, X_test, train_indices, test_indices
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test, train_indices, test_indices
    end
end

function mask_input(X::Matrix, mask_ratio::Float64, mask_val, mask_id, offset::Bool=false)
    X_masked = copy(X)
    if typeof(mask_val) == Float32
        mask_labels = fill(Float32(mask_val), size(X)) 
    else
        mask_labels = fill(Int32(mask_val), size(X)) 
    end
    n_rows, n_samples = size(X)

    if offset
        idx = 2:n_rows
        num_masked = ceil(Int, (n_rows - 1) * mask_ratio)
    else
        idx = 1:n_rows
        num_masked = ceil(Int, n_rows * mask_ratio)
    end
    
    for j in 1:n_samples
        mask_pos = shuffle(idx)[1:num_masked]
        for pos in mask_pos
            mask_labels[pos, j] = X[pos, j]
            X_masked[pos, j] = mask_id 
        end
    end
    
    return X_masked, mask_labels
end

function get_rank(value, reference_col)
    return count(x -> x > value, reference_col) + 1
end

function convert_exp_to_rank(X_test, all_trues, all_preds, all_column_indices)
    reference_matrix = X_test 
    ranked_preds = similar(all_preds, Int)
    ranked_trues = similar(all_trues, Int)

    for i in 1:length(all_preds)
        pred = all_preds[i]
        true_val = all_trues[i]
        col_idx = all_column_indices[i]
        reference_col = reference_matrix[:, col_idx]
        ranked_preds[i] = get_rank(pred, reference_col)
        ranked_trues[i] = get_rank(true_val, reference_col)
    end
    return ranked_trues, ranked_preds
end

function batch_pca(masked_idx, pca_train, mask_id)
    x = Float32.(masked_idx)
    x[x .== mask_id] .= 0
    pcatok = MultivariateStats.predict(pca_train, x)

    # avg = mean(pcatok, dims=2)
    # stddev = std(pcatok, dims=2)
    # pcatok = (pcatok .- avg) ./ (stddev .+ 1f-5)

    return pcatok
end
