using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")

using JLD2, Dates, DataFrames, Statistics, CUDA, Flux, CairoMakie, StatsBase

function plot_loss(n_epochs, train_losses, test_losses, save_dir, loss::String, pt1::Bool=false, pt2::Bool=false)
    fig_loss = Figure(size = (800, 600))
    ax_loss = Axis(fig_loss[1, 1], xlabel="epoch", ylabel="loss ($loss)", title="train vs. test loss")
    lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
    lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
    axislegend(ax_loss, position=:rt)
    
    if pt1 == true
        save(joinpath(save_dir, "pt1_loss.png"), fig_loss)
    elseif pt2 == true
        save(joinpath(save_dir, "pt2_loss.png"), fig_loss)
    else
        save(joinpath(save_dir, "loss.png"), fig_loss)
    end
end

function plot_rank_error(n_epochs, test_rank_errors, save_dir)
    fig_err = Figure(size = (800, 600))
    ax_err = Axis(fig_err[1, 1], xlabel="epoch", ylabel="error", title="mean rank errors")
    lines!(ax_err, 1:n_epochs, test_rank_errors, label="test error", linewidth=2)
    save(joinpath(save_dir, "error.png"), fig_err)
end

function plot_boxplot(n_classes, all_trues, all_preds, save_dir)
    bin_size = 50
    bin_edges = collect(1:bin_size:n_classes)
    if bin_edges[end] < n_classes
        push!(bin_edges, n_classes + 1)
    end
    bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    grouped_preds = Int[]
    grouped_trues_midpts = Float64[]
    for i in 1:length(bin_edges)-1
        indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
        if !isempty(indices)
            preds_in_bin = all_preds[indices]
            midpoint = bin_midpts[i]
            append!(grouped_preds, preds_in_bin)
            append!(grouped_trues_midpts, fill(midpoint, length(preds_in_bin)))
        end
    end
    fig_box = Figure(size = (800, 600))
    ax_box = Axis(fig_box[1, 1], xlabel="true gene id", ylabel="predicted gene id", title="predicted vs true gene ids")
    boxplot!(ax_box, grouped_trues_midpts, grouped_preds, width=bin_size*0.5)
    save(joinpath(save_dir, "boxplot.png"), fig_box)
end

function plot_hexbin(all_trues, all_preds, type::String, save_dir)
    fig_hex = Figure(size = (800, 700))
    ax_hex = Axis(fig_hex[1, 1], xlabel="true $type", ylabel="predicted $type", title="predicted vs true $type density", aspect=DataAspect())
    hexplot = hexbin!(ax_hex, all_trues, all_preds)
    Colorbar(fig_hex[1, 2], hexplot, label="point count")
    save(joinpath(save_dir, "hexbin.png"), fig_hex)
end

function plot_prediction_error(all_original_ranks, all_prediction_errors, save_dir)
    fig_rank_error_scatter = Figure(size = (800, 600))
    ax_rank_error_scatter = Axis(fig_rank_error_scatter[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "prediction error", title = "prediction error by rank")
    scatter!(ax_rank_error_scatter, all_original_ranks, all_prediction_errors, markersize=4, alpha=0.3)
    save(joinpath(save_dir, "rank_vs_error_scatter.png"), fig_rank_error_scatter)
end

function plot_mean_prediction_error(all_original_ranks, all_prediction_errors, save_dir)
    df_rank_errors = DataFrame(original_rank = all_original_ranks, prediction_error = all_prediction_errors)
    avg_errors = combine(groupby(df_rank_errors, :original_rank), :prediction_error => mean => :avg_error)
    fig_rank_error_line = Figure(size = (800, 600))
    ax_rank_error_line = Axis(fig_rank_error_line[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "mean prediction error", title = "mean prediction error by rank")
    scatter!(ax_rank_error_line, avg_errors.original_rank, avg_errors.avg_error, alpha = 0.5)
    save(joinpath(save_dir, "rank_vs_avgerror_scatter.png"), fig_rank_error_line)
    return avg_errors
end

function plot_sorted_mean_prediction_error(avg_errors, all_original_ranks, all_prediction_errors, save_dir)
    sorted_indices_by_mean = load("/home/golem/scratch/chans/lincs/plots/trt_and_untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
    error_map = Dict(row.original_rank => row.avg_error for row in eachrow(avg_errors))
    sorted_mean_errors = [get(error_map, idx, 0) for idx in sorted_indices_by_mean]
    gene_ranks = 1:length(sorted_indices_by_mean)

    fig_rank_error_line = Figure(size = (800, 600))
    ax_rank_error_line = Axis(fig_rank_error_line[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "mean prediction error", title = "mean prediction error by sorted rank")
    scatter!(ax_rank_error_line, gene_ranks, sorted_mean_errors, alpha = 0.5)
    save(joinpath(save_dir, "rank_vs_avgerror_scatter.png"), fig_rank_error_line)
    return avg_errors
end

function plot_ranked_heatmap(all_trues, all_preds, save_dir, convert::Bool)
    cs = corspearman(all_trues, all_preds)
    cp = cor(all_trues, all_preds)

    sorted_trues = all_trues
    sorted_preds = all_preds

    if convert
        sorted_indices_by_mean = load("/home/golem/scratch/chans/lincsv2/plots/untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
        gene_id_to_rank_map = invperm(sorted_indices_by_mean);
        sorted_trues = gene_id_to_rank_map[all_trues];
        sorted_preds = gene_id_to_rank_map[all_preds];
    end

    bin_edges = 1:979 
    h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))
    begin
        fig_hm = Figure(size = (500, 400))
        ax_hm = Axis(fig_hm[1, 1],
            xlabel = "true rank",
            ylabel = "predicted rank"
        )

        log10_weights = log10.(h.weights .+ 1)
        hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
        text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $cp", color = :white)
        Colorbar(fig_hm[1, 2], hm, label = "count (log10)")
        display(fig_hm)
    end
    save(joinpath(save_dir, "heatmap.png"), fig_hm)
    return cs, cp
end

function plot_exp_heatmap(all_trues, all_preds, save_dir)
    begin
        fig_hex = Figure(size = (800, 700))
        ax_hex = Axis(fig_hex[1, 1],
            # backgroundcolor = to_colormap(:viridis)[1], 
            xlabel="true expression value",
            ylabel="predicted expression value"
            # title="predicted vs. true gene id density"
            # aspect=DataAspect() 
        )
        hexplot = hexbin!(ax_hex, all_trues, all_preds, cellsize = (0.06,0.06), colorscale = log10)
        # text!(ax_hex, 0, 1050, align = (:left, :top), text = "Pearson: $cp")
        Colorbar(fig_hex[1, 2], hexplot, label="point count (log10)")
        display(fig_hex)
    end
    save(joinpath(save_dir, "exp_nn_hbin.png"), fig_hex)
end
