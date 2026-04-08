using JLD2, Dates, CUDA, Flux

function log_model(model, save_dir)
    model_cpu = cpu(model)
    jldsave(joinpath(save_dir, "model_state.jld2"); model_state=Flux.state(model_cpu))
    jldsave(joinpath(save_dir, "model_object.jld2"); model=model_cpu)
end

function get_cls(m, input, input_pca)
    embedded = m.embedding(input)
    if m.pca_mode == Val(:add)
        pca_normed = m.pca_norm(m.pca_proj(input_pca))
        pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2))
        combined = embedded .+ pca_reshaped
    elseif m.pca_mode == Val(:concat)
        processed_pca = m.pca_proj(input_pca)
        pca_reshaped = reshape(processed_pca, size(processed_pca, 1), 1, size(processed_pca, 2))
        combined = cat(pca_reshaped, embedded, dims=2)
    else
        combined = embedded
    end
    
    transformed = m.transformer(m.pos_dropout(m.pos_encoder(combined)))
    return transformed[:, 1, :] 
end

function get_profile_embeddings(X, raw_data, model, P_gpu, mu_gpu, batch_size)
    all_embeddings = []
    Flux.testmode!(model)
    
    for start_idx in 1:batch_size:size(X, 2)
        end_idx = min(start_idx + batch_size - 1, size(X, 2))
        
        x_gpu = gpu(view(X, :, start_idx:end_idx))
        
        if P_gpu !== nothing
            raw_batch_gpu = gpu(view(raw_data, :, start_idx:end_idx))
            x_pca = P_gpu * (raw_batch_gpu .- mu_gpu)
        else
            x_pca = nothing
        end
        
        batch_embeddings = cpu(get_cls(model, x_gpu, x_pca))
        push!(all_embeddings, batch_embeddings)
    end
    
    return hcat(all_embeddings...) 
end

function log_info(train_indices, test_indices, profile_embeddings, n_epochs, train_losses, test_losses, all_preds, all_trues, X_test_masked, y_test_masked, X_test, save_dir)
    jldsave(joinpath(save_dir, "indices.jld2"); train_indices=train_indices, test_indices=test_indices)
    jldsave(joinpath(save_dir, "losses.jld2"); epochs = 1:n_epochs, train_losses = train_losses, test_losses = test_losses)
    jldsave(joinpath(save_dir, "predstrues.jld2"); all_preds = all_preds, all_trues = all_trues)
    jldsave(joinpath(save_dir, "masked_test_data.jld2"); X=X_test_masked, y=y_test_masked)
    jldsave(joinpath(save_dir, "test_data.jld2"); X=X_test)

    if profile_embeddings !== nothing
        jldsave(joinpath(save_dir, "profile_embeddings.jld2"); profile_embeddings=profile_embeddings)
    end
end

function log_tf_params(config, gpu_info, run_hours, run_minutes, save_dir)
    params_txt = joinpath(save_dir, "params.txt")
    open(params_txt, "w") do io
        println(io, "PARAMETERS:")
        println(io, "########### $(gpu_info)")
        for (k, v) in pairs(config)
            println(io, "$k = $v")
        end
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    end
end