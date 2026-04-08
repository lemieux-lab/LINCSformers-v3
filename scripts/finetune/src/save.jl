using CairoMakie, JLD2, Flux

function plot_loss(epochs, train_losses, test_losses, filepath)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel="epoch", ylabel="loss (logit-ce)", title="train vs. test loss")
    lines!(ax, 1:epochs, train_losses, label="train loss", linewidth=2)
    lines!(ax, 1:epochs, test_losses, label="test loss", linewidth=2)
    axislegend(ax, position=:rt)
    CairoMakie.save(filepath, fig) 
end

function log_model(model, save_dir)
    model_cpu = cpu(model)
    jldsave(joinpath(save_dir, "model_state.jld2"), model_state=Flux.state(model_cpu))
    jldsave(joinpath(save_dir, "model_object.jld2"), model=model_cpu)
end

function save_run(save_dir, model, epochs, train_idx, test_idx, train_loss, test_loss, preds, trues; prefix="")
    log_model(model, save_dir)
    plot_loss(epochs, train_loss, test_loss, joinpath(save_dir, "$(prefix)loss.png"))   

    jldsave(joinpath(save_dir, "$(prefix)losses.jld2"); 
            epochs = 1:epochs, train_losses = train_loss, test_losses = test_loss)
    jldsave(joinpath(save_dir, "$(prefix)predstrues.jld2"); 
            all_preds = preds, all_trues = trues)
    
    idx_path = joinpath(save_dir, "indices.jld2")
    if !isfile(idx_path)
        jldsave(idx_path; train_indices = train_idx, test_indices = test_idx)
    end
end

function log_params(save_dir; kwargs...)
    open(joinpath(save_dir, "params.txt"), "w") do io
        println(io, "PARAMETERS:\n###########")
        for (key, val) in kwargs
            println(io, "$(replace(String(key), "_" => " ")) = $val") 
        end
    end
end