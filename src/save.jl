using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")

using JLD2, Dates, DataFrames, Statistics, CUDA, Flux

function log_model(model, save_dir)
    model_cpu = cpu(model)
    model_state = Flux.state(model_cpu)

    jldsave(joinpath(save_dir, "model_state.jld2"); 
    model_state=model_state
    )
    jldsave(joinpath(save_dir, "model_object.jld2"); 
        model=model_cpu
    )
end

function get_cls(m, input)
    transformed = m.transformer(m.pos_dropout(m.pos_encoder(m.embedding(input))))
    return transformed[:, 1, :] 
end

function get_profile_embeddings(X, model)
    all_embeddings = []
    Flux.testmode!(model)
    for start_idx in 1:batch_size:size(X, 2)
        end_idx = min(start_idx + batch_size - 1, size(X, 2))
        input_batch = gpu(X[:, start_idx:end_idx])
        batch_embeddings = cpu(get_cls(model, input_batch))
        push!(all_embeddings, batch_embeddings)
    end
    final_embeddings = hcat(all_embeddings...) 
    return final_embeddings
end

function log_info(train_indices, test_indices, profile_embeddings, n_epochs, 
                    train_losses, test_losses, all_preds, all_trues, 
                    all_original_ranks, all_prediction_errors, 
                    avg_errors, X_test_masked, y_test_masked, X_test)
    
    jldsave(joinpath(save_dir, "indices.jld2"); 
        train_indices=train_indices, 
        test_indices=test_indices
    )
    jldsave(joinpath(save_dir, "losses.jld2"); 
        epochs = 1:n_epochs, 
        train_losses = train_losses, 
        test_losses = test_losses
    )
    jldsave(joinpath(save_dir, "predstrues.jld2"); 
        all_preds = all_preds, 
        all_trues = all_trues
    )
    jldsave(joinpath(save_dir, "masked_test_data.jld2"); 
        X=X_test_masked, 
        y=y_test_masked
    )
    jldsave(joinpath(save_dir, "test_data.jld2"); 
        X=X_test
    )

    if profile_embeddings !== nothing
        jldsave(joinpath(save_dir, "profile_embeddings.jld2"); 
            profile_embeddings=profile_embeddings
        )
        jldsave(joinpath(save_dir, "rank_vs_error.jld2"); 
        original_rank = all_original_ranks, 
        prediction_error = all_prediction_errors
        )
        if avg_errors !== nothing
            jldsave(joinpath(save_dir, "avg_errors.jld2"); 
                original_rank = avg_errors.original_rank,
                avg_error = avg_errors.avg_error
            )
        end
    end

end

function log_tf_params(gpu_info, dataset_note, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)
    params_txt = joinpath(save_dir, "params.txt")
    open(params_txt, "w") do io
        println(io, "PARAMETERS:")
        println(io, "########### $(gpu_info)")
        println(io, "dataset = $dataset_note")
        println(io, "masking_ratio = $mask_ratio")
        println(io, "batch_size = $batch_size")
        println(io, "n_epochs = $n_epochs")
        println(io, "embed_dim = $embed_dim")
        println(io, "hidden_dim = $hidden_dim")
        println(io, "n_heads = $n_heads")
        println(io, "n_layers = $n_layers")
        println(io, "learning_rate = $lr")
        println(io, "dropout_probability = $drop_prob")
        println(io, "ADDITIONAL NOTES: $additional_notes")
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    end
end

function log_ae_params(gpu_info, dataset_note, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)
    params_txt = joinpath(save_dir, "params.txt")
    open(params_txt, "w") do io
        println(io, "PARAMETERS:")
        println(io, "########### $(gpu_info)")
        println(io, "dataset = $dataset_note")
        println(io, "masking_ratio = $mask_ratio")
        println(io, "batch_size = $batch_size")
        println(io, "n_epochs = $n_epochs")
        println(io, "latent_1 = $latent_1")
        println(io, "latent_2 = $latent_2")
        println(io, "latent_3 = $latent_3")
        println(io, "embed_dim = $embed_dim")
        println(io, "learning_rate = $lr")
        println(io, "dropout_probability = $drop_prob")
        println(io, "ADDITIONAL NOTES: $additional_notes")
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    end
end