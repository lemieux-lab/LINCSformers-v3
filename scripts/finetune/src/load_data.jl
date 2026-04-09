using Random, StatsBase, Flux, MultivariateStats, Statistics, JLD2

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

function split_data(X, test_ratio::Float64, y=nothing)
    n = size(X, 2)
    idx = shuffle(1:n)
    test_n = floor(Int, n * test_ratio)
    tst, trn = idx[1:test_n], idx[test_n+1:end]
    
    if isnothing(y)
        return X[:, trn], X[:, tst], trn, tst
    end
    return X[:, trn], y[trn], X[:, tst], y[tst], trn, tst
end

function get_labels(data::Lincs, level::String)
    X = data.expr
    if level == "lvl1"
        return X, data.inst.cell_mfc_name, 1:size(X, 2)
    elseif level == "lvl2"
        y = data.inst.pert_id
        counts = countmap(y)
        valid_labels = Set(k for (k, v) in counts if 1000 < v < 20000)
        idx = findall(l -> l in valid_labels, y)
        return X[:, idx], y[idx], idx
    else
        error("level '$level' undefined")
    end
end

function dsplit(data::Lincs, config::Config)
    X, y, orig_idx = get_labels(data, config.level)
    
    labels = unique(y)
    n_classifications = length(labels)
    ids = Dict(l => i for (i, l) in enumerate(labels))
    y_oh = Flux.onehotbatch([ids[l] for l in y], 1:n_classifications)
    n_genes = size(X, 1) 

    X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
    y_train, y_test = y_oh[:, train_indices], y_oh[:, test_indices]

    pca_info = nothing 
    if config.modeltype != "mlp"
        gene_medians = vec(median(X, dims=2)) .+ 1e-10
        X_ranked = rank_genes(X, gene_medians)

        pt_indices = load(joinpath(config.model_dir, "indices.jld2"))
        idx_dict = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(orig_idx))
        
        train_indices = filter!(x -> !isnothing(x), [get(idx_dict, i, nothing) for i in pt_indices["train_indices"]])
        test_indices  = filter!(x -> !isnothing(x), [get(idx_dict, i, nothing) for i in pt_indices["test_indices"]])

        X_train, X_test = X_ranked[:, train_indices], X_ranked[:, test_indices]
        y_train, y_test = y_oh[:, train_indices], y_oh[:, test_indices]

        if config.modeltype in ("v1", "v2")
            raw_train = X[:, train_indices]
            raw_test = X[:, test_indices]
            pca_train_norm = fit(PCA, Float32.(data.expr[:, pt_indices["train_indices"]]); maxoutdim=config.embed_dim)
            pca_info = (norm=pca_train_norm, raw_train=raw_train, raw_test=raw_test)
        end
    end

    cidx_dict, cs = (config.level == "lvl2") ? oversmpl(y_train) : (nothing, nothing)

    return X_train, X_test, y_train, y_test, train_indices, test_indices, n_genes, n_classifications, cidx_dict, cs, pca_info
end

function get_pooled(m, x, x_pca, m_type)
    embedded = m.embedding(x)
    
    if m_type == "rtf"
        encoded = m.pos_encoder(embedded)
    elseif m_type == "v1"
        processed_pca = m.use_pca_proj ? m.pca_proj(x_pca) : x_pca
        pca_reshaped = reshape(processed_pca, size(processed_pca, 1), 1, size(processed_pca, 2))
        combined = cat(pca_reshaped, embedded, dims=2)
        encoded = m.pos_encoder(combined)
    elseif m_type == "v2"
        pca_embedded = m.pca_proj(x_pca)
        pca_normed = m.pca_norm(pca_embedded)
        pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2))
        combined = embedded .+ pca_reshaped
        encoded = m.pos_encoder(combined)
    end
    
    encoded_dropped = m.pos_dropout(encoded)
    transformed = m.transformer(encoded_dropped)
    return dropdims(mean(transformed, dims=2), dims=2)
end

function emb(config::Config, X_train, X_test, pca_info, use_pca, n_genes, n_classifications)
    if config.modeltype == "mlp"
        ft_model = Flux.Chain(
            Flux.Dense(n_genes => config.hidden_dim, gelu),
            Flux.LayerNorm(config.hidden_dim),
            Flux.Dropout(config.drop_prob),
            Flux.Dense(config.hidden_dim => n_classifications)
        ) |> gpu
        return ft_model, Float32.(X_train), Float32.(X_test)
    end

    state = load(joinpath(config.model_dir, "model_state.jld2"))["model_state"]
    general_model = (
        input_size=n_genes + 1, embed_dim=config.embed_dim, n_layers=config.n_layers,
        n_classes=n_genes, n_heads=config.n_heads, hidden_dim=config.hidden_dim, dropout_prob=config.drop_prob
    )

    if config.modeltype == "rtf"
        pt_model = Model(; general_model...) |> gpu
    elseif config.modeltype == "v1"
        pt_model = Model(; general_model..., pca_dim=config.embed_dim, use_pca_proj=false) |> gpu
    elseif config.modeltype == "v2"
        pt_model = Model(; general_model..., pca_dim=config.embed_dim) |> gpu
    end
    
    Flux.loadmodel!(pt_model, state)
    Flux.testmode!(pt_model)

    function extract_embeds(X_ranked, raw_data)
        embeds = []
        n_samples = size(X_ranked, 2)
        for i in 1:config.batch_size:n_samples
            b_idx = i:min(i+config.batch_size-1, n_samples)
            x_batch = gpu(Int32.(X_ranked[:, b_idx]))
            
            if use_pca && !isnothing(pca_info)
                x_pca_batch = gpu(Float32.(MultivariateStats.predict(pca_info.norm, Float32.(raw_data[:, b_idx]))))
                push!(embeds, cpu(get_pooled(pt_model, x_batch, x_pca_batch, config.modeltype)))
            else
                push!(embeds, cpu(get_pooled(pt_model, x_batch, nothing, config.modeltype)))
            end
        end
        return hcat(embeds...)
    end

    train_input = extract_embeds(X_train, use_pca ? pca_info.raw_train : nothing)
    test_input = extract_embeds(X_test, use_pca ? pca_info.raw_test : nothing)

    ft_model = Flux.Chain(
        Flux.Dense(config.embed_dim => config.hidden_dim, gelu),
        Flux.LayerNorm(config.hidden_dim),
        Flux.Dropout(config.drop_prob),
        Flux.Dense(config.hidden_dim => n_classifications)
    ) |> gpu

    return ft_model, train_input, test_input
end

function mstate(config::Config, pca_info, use_pca, n_classifications)
    state = load(joinpath(config.model_dir, "model_state.jld2"))["model_state"]
    
    general_model = (
        input_size=979, embed_dim=config.embed_dim, n_layers=config.n_layers,
        n_classes=978, n_heads=config.n_heads, hidden_dim=config.hidden_dim, dropout_prob=config.drop_prob
    )

    if config.modeltype == "rtf"
        pt_model = Model(; general_model...) |> gpu
    elseif config.modeltype == "v1"
        pt_model = Model(; general_model..., pca_dim=config.embed_dim, use_pca_proj=false) |> gpu
    elseif config.modeltype == "v2"
        pt_model = Model(; general_model..., pca_dim=config.embed_dim) |> gpu
    end

    Flux.loadmodel!(pt_model, state)

    ft_model = FTModel(pt_model;
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_classifications=n_classifications
    ) |> gpu

    if use_pca && !isnothing(pca_info)
        X_pca_train = Float32.(MultivariateStats.predict(pca_info.norm, Float32.(pca_info.raw_train)))
        X_pca_test  = Float32.(MultivariateStats.predict(pca_info.norm, Float32.(pca_info.raw_test)))
    else
        X_pca_train = nothing
        X_pca_test = nothing
    end
    
    return ft_model, X_pca_train, X_pca_test
end

function oversmpl(y_train)
    labels = Flux.onecold(cpu(y_train))
    cidx_dict = Dict{Int, Vector{Int}}()
    for (i, label) in enumerate(labels)
        push!(get!(cidx_dict, label, Int[]), i)
    end
    return cidx_dict, collect(keys(cidx_dict))
end