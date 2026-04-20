using Flux, ProgressBars, Statistics

function manual_logitcrossentropy(logits, y; dims=1)
    max_logits = maximum(logits, dims=dims)
    shifted = logits .- max_logits
    log_sum_exp = log.(sum(exp.(shifted), dims=dims))
    log_probs = shifted .- log_sum_exp
    return -mean(sum(y .* log_probs, dims=dims))
end

function ce_loss(model, x, x_pca, y, use_pca)
    logits = use_pca ? model(x, x_pca) : model(x)
    return manual_logitcrossentropy(logits, y)
end

function train(model, opt, data, config, logs)
    (; X_train, ytrain, X_test, y_test, pca_train, pca_test) = data
    (; epochs, batch_size, loss, use_pca, use_oversmpl, clsdict, cls, freq, save_dir, pt) = config

    for epoch in ProgressBar(1:epochs)
        println("epoch $epoch")
        is_last = (epoch == epochs)
        # is_checkpt = !isnothing(freq) && (epoch % freq == 0 || is_last)
        is_checkpt = (!isnothing(freq) && epoch % freq == 0) || is_last

        train_loss = train_epoch!(model, opt, X_train, ytrain, pca_train, loss, 
                                  use_oversmpl, clsdict, cls, use_pca, batch_size)
        push!(logs.train_losses, train_loss)

        test_loss, epoch_preds, epoch_trues = eval_epoch(model, X_test, y_test, pca_test, 
                                                         use_pca, batch_size, is_checkpt)
        push!(logs.test_losses, test_loss)
    
        if is_last
            append!(logs.preds, epoch_preds)
            append!(logs.trues, epoch_trues)
        end

        if is_checkpt
            path = joinpath(save_dir, "$(pt)_epoch_$(epoch).jld2")
            jldsave(path; 
                model_state = Flux.state(cpu(model)),
                train_losses = logs.train_losses, test_losses = logs.test_losses, 
                checkpt_preds = epoch_preds, checkpt_trues = epoch_trues, 
                completed_epoch = epoch
            )
        end
    end
end

function train_epoch!(model, opt, X_train, ytrain, pca_train, loss, 
                      use_oversmpl, clsdict, cls, use_pca, batch_size)
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    n_samples = size(X_train, 2)
    num_batches = div(n_samples, batch_size)
    
    for i in 1:num_batches
        if use_oversmpl
            batch_idx = [rand(clsdict[rand(cls)]) for _ in 1:batch_size]
        else
            start_idx = (i - 1) * batch_size + 1
            end_idx = min(start_idx + batch_size - 1, n_samples)
            batch_idx = start_idx:end_idx
        end

        x_gpu = cu(X_train[:, batch_idx])
        y_gpu = cu(ytrain[:, batch_idx])
        x_pca = use_pca ? cu(pca_train[:, batch_idx]) : nothing

        lv, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, x_pca, y_gpu, use_pca)
        end
        # lv = loss(model, x_gpu, x_pca, y_gpu, use_pca)
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, Float32(cpu(lv)))
    end
    return mean(epoch_losses)
end

function eval_epoch(model, X_test, y_test, pca_test, use_pca, batch_size, get_preds)
    Flux.testmode!(model)
    epoch_losses = Float32[]
    epoch_preds, epoch_trues = Int[], Int[]
    n_samples = size(X_test, 2)

    for start_idx in 1:batch_size:n_samples
        end_idx = min(start_idx + batch_size - 1, n_samples)
        batch_idx = start_idx:end_idx

        x_gpu = cu(X_test[:, batch_idx])
        y_gpu = cu(y_test[:, batch_idx])
        x_pca = use_pca ? cu(pca_test[:, batch_idx]) : nothing

        logits = use_pca ? model(x_gpu, x_pca) : model(x_gpu)
        push!(epoch_losses, Float32(cpu(Flux.logitcrossentropy(logits, y_gpu))))

        if get_preds
            append!(epoch_preds, Flux.onecold(cpu(logits)))
            append!(epoch_trues, Flux.onecold(cpu(y_gpu)))
        end
    end
    return mean(epoch_losses), epoch_preds, epoch_trues
end

function manual_gpu_transfer(x)
    if x isa Flux.Dropout
        return Flux.Dropout(x.p; dims=x.dims, rng=CUDA.default_rng())
    elseif x isa AbstractArray
        return cu(x)
    else
        return x
    end
end