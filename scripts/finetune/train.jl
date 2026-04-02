using Flux, ProgressBars, MultivariateStats, Statistics

function ce_loss(model, x, x_pca, y, use_pca)
    if use_pca
        logits = model(x, x_pca)
    else
        logits = model(x)
    end
    return Flux.logitcrossentropy(logits, y)
end

function train(ft_model, opt, xtrain, ytrain, xtest, ytest,
    epochs, train_losses, test_losses, preds, trues, loss_fn, 
    use_oversampling, class_to_idx, avail_classes, use_pca, batch_size,
    xpca_train=nothing, xpca_test=nothing
    )
    for epoch in ProgressBar(1:epochs)

        iter = div(size(X_train, 2), batch_size)
        
        Flux.trainmode!(ft_model)
        train_epoch_losses = Float32[]
        
        # for start_idx in 1:batch_size:size(X_train, 2)
            # end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
        for i in 1:iter

            # x_gpu = gpu(Int32.(X_train[:, start_idx:end_idx]))
            # y_gpu = gpu(Float32.(y_train[:, start_idx:end_idx]))

            if use_oversampling
                batch_idx = oversample_batch(class_to_idx, avail_classes, batch_size)
            else
                start_idx = (i - 1) * batch_size + 1
                end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
                batch_idx = start_idx:end_idx
            end

            x_gpu = gpu(xtrain[:, batch_idx])
            y_gpu = gpu(ytrain[:, batch_idx])

            if use_pca
                # raw_batch_cpu = raw_train_norm[:, batch_idx]
                # x_pca_cpu = MultivariateStats.predict(pca_train_norm, raw_batch_cpu)
                # x_pca = gpu(Float32.(x_pca_cpu))

                x_pca = gpu(xpca_train[:, batch_idx])

            else
                x_pca = nothing
            end

            lv, grads = Flux.withgradient(ft_model) do m
                loss_fn(m, x_gpu, x_pca, y_gpu, use_pca)
            end
            Flux.update!(opt, ft_model, grads[1])

            lv = loss_fn(ft_model, x_gpu, x_pca, y_gpu, use_pca)

            push!(train_epoch_losses, lv) 
        end
        push!(train_losses, mean(train_epoch_losses))

        Flux.testmode!(ft_model)
        test_epoch_losses = Float32[]

        for start_idx in 1:batch_size:size(xtest, 2)
            end_idx = min(start_idx + batch_size - 1, size(xtest, 2))

            x_gpu = gpu(xtest[:, start_idx:end_idx])
            y_gpu = gpu(ytest[:, start_idx:end_idx])

            if use_pca
                # raw_batch_cpu = raw_test_norm[:, start_idx:end_idx]
                # x_pca_cpu = MultivariateStats.predict(pca_train_norm, raw_batch_cpu)
                # x_pca = gpu(Float32.(x_pca_cpu))
                x_pca = gpu(xpca_test[:, start_idx:end_idx])
                logits = ft_model(x_gpu, x_pca)
            else
                logits = ft_model(x_gpu)
            end

            test_lv = Flux.logitcrossentropy(logits, y_gpu)
            push!(test_epoch_losses, test_lv)

            if epoch == epochs
                batch_preds = Flux.onecold(cpu(logits))
                batch_trues = Flux.onecold(cpu(y_gpu))
                append!(preds, batch_preds)
                append!(trues, batch_trues)
            end
        end
        push!(test_losses, mean(test_epoch_losses))
    end
end