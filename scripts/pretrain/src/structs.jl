using Flux

struct PosEnc{A<:AbstractArray}
    pe_matrix::A
end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        pe_matrix[i,pos] = mod(i, 2) == 1 ? sin(angle) : cos(angle)
    end
    return PosEnc(pe_matrix)
end

Flux.@layer PosEnc
Flux.trainable(pe::PosEnc) = NamedTuple()
(pe::PosEnc)(input::AbstractArray) = input .+ @view(pe.pe_matrix[:, 1:size(input,2)])

struct Transf{A,D,N,M}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M 
    mlp_norm::N
end

function Transf(embed_dim::Int, hidden_dim::Int; n_heads::Int, dropout_prob::Float64)
    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, nheads=n_heads, dropout_prob=dropout_prob)
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu), Flux.Dropout(dropout_prob), Flux.Dense(hidden_dim => embed_dim), Flux.Dropout(dropout_prob))
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@layer Transf

function (tf::Transf)(input)
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1]
    residualed = input + tf.att_dropout(atted)
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size)
    mlp_out_reshaped = reshape(tf.mlp(reshaped), embed_dim, seq_len, batch_size)
    return residualed + mlp_out_reshaped
end

struct Model{E,J,N,P,D,T,C,M}
    embedding::E
    pca_proj::J
    pca_norm::N
    pos_encoder::P
    pos_dropout::D
    transformer::T
    classifier::C
    pca_mode::Val{M}
end

Flux.@layer Model

function Model(; input_size::Int, pca_dim::Int=0, embed_dim::Int, n_layers::Int, n_classes::Int, n_heads::Int, hidden_dim::Int, dropout_prob::Float64, pca_mode::Symbol)
    embedding = Flux.Embedding(input_size => embed_dim)
    pca_proj = pca_mode == :none ? identity : Flux.Dense(pca_dim => embed_dim)
    pca_norm = pca_mode == :add ? Flux.LayerNorm(embed_dim) : identity
    pe_len = pca_mode == :concat ? input_size + 1 : input_size 
    pos_encoder = PosEnc(embed_dim, pe_len)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    classifier = Flux.Chain(Flux.Dense(embed_dim => embed_dim, gelu), Flux.LayerNorm(embed_dim), Flux.Dense(embed_dim => n_classes))
    return Model(embedding, pca_proj, pca_norm, pos_encoder, pos_dropout, transformer, classifier, Val(pca_mode))
end

(m::Model)(x, x_pca=nothing) = m(x, x_pca, m.pca_mode)

function (model::Model)(input, input_pca, ::Val{:concat})
    embedded = model.embedding(input)
    processed_pca = model.pca_proj(input_pca)
    pca_reshaped = reshape(processed_pca, size(processed_pca, 1), 1, size(processed_pca, 2))
    combined = cat(pca_reshaped, embedded, dims=2)
    encoded = model.transformer(model.pos_dropout(model.pos_encoder(combined)))
    logits = model.classifier(encoded)
    return logits[:, 2:end, :] 
end

function (model::Model)(input, input_pca, ::Val{:add})
    embedded = model.embedding(input)
    pca_normed = model.pca_norm(model.pca_proj(input_pca))
    pca_reshaped = reshape(pca_normed, size(pca_normed, 1), 1, size(pca_normed, 2))
    combined = embedded .+ pca_reshaped 
    encoded = model.transformer(model.pos_dropout(model.pos_encoder(combined)))
    return model.classifier(encoded)
end

function (model::Model)(input, input_pca, ::Val{:none})
    embedded = model.embedding(input)
    encoded = model.transformer(model.pos_dropout(model.pos_encoder(embedded)))
    return model.classifier(encoded)
end