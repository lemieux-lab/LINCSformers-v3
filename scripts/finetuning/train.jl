using Flux, ProgressBars

function ce_loss(model, x, y)
    logits = model(x)
    return Flux.logitcrossentropy(logits, y)
end

function train(epochs, train_losses, test_losses, preds, trues, loss)
    for epoch in ProgressBar(1:epochs)
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
        push!(train_losses, mean(train_epoch_losses))

        test_epoch_losses = Float32[]

        for start_idx in 1:batch_size:size(X_test, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_test, 2))

            x_gpu = gpu(Int32.(X_test[:, start_idx:end_idx]))
            y_gpu = gpu(Float32.(y_test[:, start_idx:end_idx]))

            test_loss_val = loss(ft_model, x_gpu, y_gpu)
            push!(test_epoch_losses, test_loss_val)

            logits = ft_model(x_gpu)
            test_loss_val = Flux.logitcrossentropy(logits, y_gpu)

            if epoch == epochs
                preds = Flux.onecold(cpu(logits))
                trues = Flux.onecold(cpu(y_gpu))
                append!(preds, preds)
                append!(trues, trues)
            end
        end
        push!(test_losses, mean(test_epoch_losses))
    end
end