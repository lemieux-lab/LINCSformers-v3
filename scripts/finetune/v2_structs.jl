using Pkg
Pkg.activate("/home/golem/scratch/chans/lincsv3")
Pkg.instantiate()

using LincsProject, JLD2, Flux, Optimisers, ProgressBars, Statistics, CUDA, Dates

include("../../src/params.jl")
include("../../src/fxns.jl")
include("../../src/plot.jl")
include("../../src/save.jl")


# pretrained model struct for reconstruction


### positional encoder
struct PosEnc{A<:AbstractArray}
    pe_matrix::A
end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(pe_matrix)
end

Flux.@layer PosEnc
Flux.trainable(pe::PosEnc) = NamedTuple()

function (pe::PosEnc)(input::AbstractArray)
    seq_len = size(input,2)
    return input .+ @view(pe.pe_matrix[:, 1:seq_len]) # adds positional encoding to input embeddings
end


struct Transf{A,D,N,M}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M 
    mlp_norm::N
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )
    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@layer Transf

function (tf::Transf)(input) # input shape: embed_dim × seq_len × batch_size
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)

    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped

    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped

    return tf_output
end


struct Model{E,J,N,P,D,T,C}
    embedding::E
    pca_proj::J
    pca_norm::N
    # cls_token::L
    pos_encoder::P
    pos_dropout::D
    transformer::T
    classifier::C
end

function Model(;
    input_size::Int,
    pca_dim::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )
    embedding = Flux.Embedding(input_size => embed_dim)
    pca_proj = Flux.Dense(pca_dim => embed_dim)
    pca_norm = Flux.LayerNorm(embed_dim)
    # cls_token = Flux.Embedding(1 => embed_dim)
    pos_encoder = PosEnc(embed_dim, input_size)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )
    return Model(embedding, pca_proj, pca_norm, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@layer Model

function (model::Model)(input, input_pca) # input: n,b input_pca: p,b
    embedded = model.embedding(input) # e,n,b
    pca_embedded = model.pca_proj(input_pca) # e,b
    # cls_embedded = model.cls_token.weight # e,1

    # hybrid_emb = pca_embedded .+ cls_embedded # e,b
    # hybrid_reshaped = reshape(hybrid_emb, size(hybrid_emb, 1), 1, size(hybrid_emb, 2)) # e,1,b

    # pca_reshaped = reshape(pca_embedded, size(pca_embedded, 1), 1, size(pca_embedded, 2))
    # combined = cat(hybrid_reshaped, embedded, dims=2) # e,n+1,b

    pca_normed = model.pca_norm(pca_embedded)

    pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2)) # e,1,b
    # pca_reshaped = reshape(pca_embedded, size(pca_embedded, 1), 1, size(pca_embedded, 2)) # e,1,b
    combined = embedded .+ pca_reshaped # e,n,b

    encoded = model.pos_encoder(combined) # e,n+1,b
    encoded_dropped = model.pos_dropout(encoded) # e,n+1,b

    transformed = model.transformer(encoded_dropped) # e,n+1,b

    # cls = transformed[:,1,:]
    # logits_output = model.classifier(cls)

    logits_output = model.classifier(transformed) # n,n+1,b
    
    return logits_output
end


# finetuned struct to stick them together


struct FTModel{P,H}
    pretrained::P
    head::H
end

Flux.@layer FTModel

function FTModel(pt_model; 
    embed_dim::Int, 
    hidden_dim::Int, 
    n_classifications::Int
    )
    
    pretrained = (
        embedding = pt_model.embedding,
        pca_proj = pt_model.pca_proj,
        pca_norm = pt_model.pca_norm,
        pos_encoder = pt_model.pos_encoder,
        pos_dropout = pt_model.pos_dropout,
        transformer = pt_model.transformer
    )
    
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications)
    )
    
    return FTModel(pretrained, head)
end

function (m::FTModel)(input, input_pca)
    embedded = m.pretrained.embedding(input)
    pca_embedded = m.pretrained.pca_proj(input_pca)
    pca_normed = m.pretrained.pca_norm(pca_embedded)
    
    pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2)) 
    combined = embedded .+ pca_reshaped 
    encoded = m.pretrained.pos_encoder(combined)
    encoded_dropped = m.pretrained.pos_dropout(encoded)
    transformed = m.pretrained.transformer(encoded_dropped)
    pooled = dropdims(mean(transformed, dims=2), dims=2)
    return m.head(pooled)
end