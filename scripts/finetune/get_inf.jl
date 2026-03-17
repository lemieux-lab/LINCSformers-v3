using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, HDF5, CSV, DataFrames

# holy moly this is for level 3 downstream task

function get_inferred(prefix::String, gctx::String, filtered_data::Lincs, out_file::String)
    
    f = h5open("$prefix/$gctx")
    ptr_mat = f["0/DATA/0/matrix"] # pointer to gene expression vals
    gene_ids = String.(f["0/META/ROW/id"][:])
    sample_ids = String.(f["0/META/COL/id"][:]) 

    gene_df = CSV.File("$prefix/geneinfo_beta.txt", delim="\t", types=String) |> DataFrame
    inf_df = filter(row -> row.feature_space != "landmark", gene_df)
    inf_id = String.(inf_df.gene_id)
    inf_row_idx = [findfirst(id -> id == sym, gene_ids) for sym in inf_id]

    target_sample_ids = String.(filtered_data.inst.sample_id)
    col_dict = Dict(val => i for (i, val) in enumerate(sample_ids))
    target_col_idx = [col_dict[s] for s in target_sample_ids]

    n_inf = length(inf_row_idx)
    n_samples = length(target_col_idx)
    y_target = Matrix{Float32}(undef, n_inf, n_samples)

    for (i, col_idx) in enumerate(target_col_idx)
        full_col = ptr_mat[:, col_idx] 
        y_target[:, i] = full_col[inf_row_idx] 
    end
    
    close(f)
    jldsave(out_file; y_target, inf_df)
    
    return y_target
end

dir = "/home/golem/scratch/chans/lincs/data"
loading_dir = "$dir/lincs_loading_files"
lvl3_gctx = "level3_beta_all_n3026460x12328.gctx"

# for untrt only 

data_path = "/home/golem/scratch/chans/lincsv3/data/lincs_untrt_data.jld2"
untrt_data = load(data_path)["filtered_data"]
untrt_out = "$dir/lincs_untrt_inferred_data.jld2"

y_inf = get_inferred(loading_dir, lvl3_gctx, untrt_data, untrt_out)


# for trt and untrt

data_path = "/home/golem/scratch/chans/lincsv3/data/lincs_trt_untrt_data.jld2"
trt_untrt_data = load(data_path)["filtered_data"]
trt_untrt_out = "$dir/lincs_trt_untrt_inferred_data.jld2"

y_inf = get_inferred(loading_dir, lvl3_gctx, trt_untrt_data, trt_untrt_out)