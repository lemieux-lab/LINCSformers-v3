# for reformatting heatmaps/hexbins after running model
# TODO: need to clean this up a bit

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")

using JLD2, CairoMakie, StatsBase, StatisticalMeasures, CategoricalArrays

dir = "/home/golem/scratch/chans/lincsv3/plots/trt/rtf_v1/2026-03-11_10-42"
all_trues = load("$dir/predstrues.jld2")["all_trues"]
all_preds = load("$dir/predstrues.jld2")["all_preds"]

cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)



# for exp val

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
save_dir = "/home/golem/scratch/chans/lincsv2/plots/untrt/TEST_rank_tf/baseline"
save(joinpath(save_dir, "exp_nn_hbin.png"), fig_hex)
print(cs)



# for rank id

# # to sort x axis
sorted_indices_by_mean = load("/home/golem/scratch/chans/lincsv2/plots/untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
gene_id_to_rank_map = invperm(sorted_indices_by_mean);
sorted_trues = gene_id_to_rank_map[all_trues];
sorted_preds = gene_id_to_rank_map[all_preds];

bin_edges = 1:979 
h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))
begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp, digits=4))", color = :white)
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save("$dir/hmap.png", fig_hm)



# for rank nn

rank_data = load("/home/golem/scratch/chans/lincsv3/plots/untrt/rank_nn/2026-01-14_15-26/rankedpredstrues.jld2")

sorted_trues = rank_data["ranked_trues"]
sorted_preds = rank_data["ranked_preds"]

cp_rank = cor(sorted_trues, sorted_preds)

bin_edges = 1:979 
h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))

begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp_rank, digits=4))", color = :white)
    
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save("$dir/hmap.png", fig_hm)



# for oh_rank_nn

dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/oh_rank_nn/2026-01-14_11-23" 
all_trues = load("$dir/predstrues.jld2")["all_trues"]
all_preds = load("$dir/predstrues.jld2")["all_preds"]

cp = cor(Float64.(all_trues), Float64.(all_preds))
max_id = maximum(vcat(all_trues, all_preds))
bin_edges = 1:(max_id + 1) 

h = fit(Histogram, (all_trues, all_preds), (bin_edges, bin_edges))

begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank",
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp, digits=4))", color = :white)
    
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save("$dir/hmap.png", fig_hm)



# for lvl 1 finetune

dir = "/home/golem/scratch/chans/lincsv3/plots/trt/finetuning/lvl1/rtf/2026-03-17_17-19"

all_trues = load("$dir/pt2_predstrues_recovered.jld2")["all_trues"]
all_preds = load("$dir/pt2_predstrues_recovered.jld2")["all_preds"]

# finetune metricsc
classes = union(unique(all_trues), unique(all_preds))
y_true = categorical(all_trues, levels=classes)
y_pred = categorical(all_preds, levels=classes)
acc = accuracy(y_pred, y_true)
f1 = macro_f1score(y_pred, y_true)
# prec = multiclass_precision(y_pred, y_true) # not really sure if we need this
cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

bin_edges = 1:306
h = fit(Histogram, (all_trues, all_preds), (bin_edges, bin_edges))
begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True cell line",
        ylabel = "Predicted cell line"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 15, 290, align = (:left, :top), text = "Acc: $(round(acc, digits=4))", color = :white)
    text!(ax_hm, 15, 260, align = (:left, :top), text = "F1: $(round(f1, digits=4))", color = :white)
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save("$dir/hmap.png", fig_hm)



# for lvl 2 finetune

dir = "/home/golem/scratch/chans/lincsv3/plots/trt/finetuning/lvl2/v1/2026-03-17_11-11"
all_trues = load("$dir/pt2_predstrues_recovered.jld2")["all_trues"]
all_preds = load("$dir/pt2_predstrues_recovered.jld2")["all_preds"]

# finetune metricsc
classes = union(unique(all_trues), unique(all_preds))
y_true = categorical(all_trues, levels=classes)
y_pred = categorical(all_preds, levels=classes)
acc = accuracy(y_pred, y_true)
f1 = macro_f1score(y_pred, y_true)
# prec = multiclass_precision(y_pred, y_true) # not really sure if we need this
cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

# true_counts = countmap(all_trues)
# correct_counts = Dict{Int, Int}()
# for (t, p) in zip(all_trues, all_preds)
#     if t == p
#         correct_counts[t] = get(correct_counts, t, 0) + 1
#     end
# end

# classes = collect(keys(true_counts))
# supports = [true_counts[c] for c in classes]
# accuracies = [get(correct_counts, c, 0) / true_counts[c] for c in classes]

# begin
#     fig = Figure(size = (400, 300))
#     ax = Axis(fig[1, 1], 
#         xlabel = "Number of True Samples in Test Set (Log Scale)", 
#         ylabel = "Accuracy (0.0 to 1.0)",
#         xscale = log10)

#     scatter!(ax, supports, accuracies, 
#         markersize = 5, 
#         color = (:darkblue, 0.4))
#     display(fig)
# end
# # save("$dir/accuracy_vs_support.png", fig)

true_counts = countmap(all_trues)
sorted_classes = sort(collect(true_counts), by=last, rev=true)
top_classes = [k for (k, v) in sorted_classes[1:101]]
class_to_idx = Dict(c => i for (i, c) in enumerate(top_classes))
mapped_trues = Int[]
mapped_preds = Int[]

for (t, p) in zip(all_trues, all_preds)
    if haskey(class_to_idx, t) && haskey(class_to_idx, p)
        push!(mapped_trues, class_to_idx[t])
        push!(mapped_preds, class_to_idx[p])
    end
end

bin_edges = 1:101 
h = fit(Histogram, (mapped_trues, mapped_preds), (bin_edges, bin_edges))
begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True perturbation rank (1 = most frequent (:DMSO))",
        ylabel = "Predicted perturbation rank",
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 5, 95, align = (:left, :top), text = "Acc: $(round(acc, digits=4))", color = :white)
    text!(ax_hm, 5, 85, align = (:left, :top), text = "F1: $(round(f1, digits=4))", color = :white)
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save("$dir/hmap.png", fig_hm)