#include "decoder.hpp"

#include <functional>

#include "mappedfile.hpp"
#include "tensor.hpp"
#include "kvcache.hpp"

using namespace decoder;
using namespace tn;
using namespace std;
using namespace kvcache;

// See https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html
inline void gelu_(Tensor& x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            const float v = x.at(i, j);
            const float inner = k * (v + 0.044715f * v * v * v);
            x.set(i, j, 0.5f * v * (1.0f + tanhf(inner)));
        }
    }
}

// input shape (T, C), output shape (T, C)
// See https://arxiv.org/pdf/2002.04745 Figure 1b.
Tensor Layer::forward(const Tensor& x) {
    Tensor out(x.rows(), x.cols());
    Tensor h(x.rows(), x.cols());
    Tensor h1(x.rows(), x.cols());
    out = ln1_.forward(x);
    out = attn_.forward(out);
    h = out + x;  // add the residual stream
    h1 = ln2_.forward(h);
    h1 = ffn_.forward(h1);
    return h + h1;  // add the residual stream
}

Tensor Layer::decode(KVCache &kv_cache, int layer_index, const Tensor& x)
{
    Tensor out(x.rows(), x.cols());
    Tensor h(x.rows(), x.cols());
    Tensor h1(x.rows(), x.cols());
    out = ln1_.forward(x);
    out = attn_.decode(kv_cache, layer_index, out);
    h = out + x;  // add the residual stream
    h1 = ln2_.forward(h);
    h1 = ffn_.forward(h1);
    return h + h1;  // add the residual stream
}

// Linear (input, hidden), GELU, Linear (hidden, output)
Tensor FFN::forward(const Tensor& x) const {
    Tensor h = l1_.forward(x);
    gelu_(h);
    return l2_.forward(h);
}

// y = x@w_T + b
// w (ouptut feature, input feature)
// b (output feature) - b is broadcasted
// x (batch dim, input feature)
// y (batch dim, output feature)
Tensor Linear::forward(const Tensor& x) const {
    if (b_.has_value()) {
        assert(static_cast<int>((*b_).size()) == w_.rows);
    }

    assert(x.cols() == w_.cols);
    Tensor y(x.rows(), w_.rows);
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < w_.rows; j++) {
            float val = 0.0f;
            for (int k = 0; k < x.cols(); k++) {
                val += (x.at(i, k) * w_.at(j, k));
            }
            if (b_.has_value()) {
                y.set(i, j, val + (*b_)[j]);
            } else {
                y.set(i, j, val);
            }
        }
    }
    return y;
}

// input matrix (A, B)
// output matrix with values in dimension B normed.
// See: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html
Tensor LayerNorm::forward(const Tensor& x) const {
    assert(x.cols() == static_cast<int>(w_.size()));
    assert(x.cols() == static_cast<int>(b_.size()));
    Tensor out(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++) {
        float mu = 0, var = 0;
        for (int j = 0; j < x.cols(); j++) {
            mu += x.at(i, j);
        }
        mu /= x.cols();
        for (int j = 0; j < x.cols(); j++) {
            var += (x.at(i, j) - mu) * (x.at(i, j) - mu);
        }
        var /= x.cols();  // Var(x) = E[(X - mu)^2], mu = E[x]
        float s = sqrt(var + eps_);
        for (int j = 0; j < x.cols(); j++) {
            out.set(i, j, (((x.at(i, j) - mu) / s) * w_[j]) + b_[j]);
        }
    }
    return out;
}

// Multi-head self-attention
// input: Shape (T, C)
// output: Shape (T, C)
// T = tokens, C = embedding dimensions
Tensor MHSA::forward(const Tensor& x) const {
    int C = x.cols();  // For gpt-2 this is 768
    int T = x.rows();
    Tensor out(T, C);  // Shape (T, C)

    // Check tensor shapes for sanity
    assert(qkvw_.size() == static_cast<size_t>(C * 3 * C));
    assert(qkvb_.size() == static_cast<size_t>(3 * C));
    assert(attprojw_.size() == static_cast<size_t>(C * C));
    assert(attprojb_.size() == static_cast<size_t>(C));

    // first step is the linear: h = x @ qkvw^T + qkvb
    // qkvw is of shape (3C, C).
    // shape: lqkv (T, 3C) <- x (T, C) @ qkvw (3C, C)^T + qkvb (3C)
    MatrixView qkvw(&qkvw_[0], 3 * C, C);

    // Now apply the linear
    Linear lqkv(qkvw, qkvb_);
    Tensor qkv = lqkv.forward(x);  // Shape (T, 3C)
    assert(qkv.rows() == T && qkv.cols() == 3 * C);

    if (use_kv_cache_)
    {
        for (int i = 0; i < T; i++)
        {
            Tensor k = qkv.slice(i, i + 1, C, 2 * C);
            Tensor v = qkv.slice(i, i + 1, 2 * C, 3 * C);

            kv_cache_.add(k, v);
        }
    }

    // next step is to separate the qkv Tensor shape (T, 3C)
    // into 3 Tensors q, k, v each shape = (T, C)
    // Also, we need to split each q, k, v further into shape (T, C/nheads) for each of the
    // attention heads. Since we only have a 2-D Tensor library, run a for loop nheads times
    // with each q, k, v matrix of shape = (T, C/nheads)
    // we'll do this with a 0-copy MatrixView that offsets cleverly into the qkv Tensor
    assert(C % nheads_ == 0);
    for (int i = 0; i < nheads_; i++) {
        // Per head q, k, and v. Shape (T, C/nheads)
        MatrixView qh(qkv.raw() + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
        MatrixView kh(qkv.raw() + C + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
        MatrixView vh(qkv.raw() + (2 * C) + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
        // Shape (C/nheads, T)
        MatrixView kht = kh.transpose();
        // scaled dot product (T, C/nheads) @ (C/nheads, T) -> (T, T)
        Tensor sdp = matmul(qh, kht).scale(1.0 / sqrt(C / nheads_));  // Shape (T, T)
        assert(sdp.rows() == T && sdp.cols() == T);
        // Apply causal mask - upper triangular entries go to -INF before softmax
        for (int sdp_row = 0; sdp_row < sdp.rows(); sdp_row++) {
            for (int sdp_col = 0; sdp_col < sdp.cols(); sdp_col++) {
                if (sdp_col > sdp_row) {
                    sdp.set(sdp_row, sdp_col, -numeric_limits<float>::infinity());
                }
            }
        }
        // Softmax
        Tensor softmax = sdp.softmax();  // Shape (T, T)
        // Extract the value vector
        Tensor val = softmax.matmul(vh);  // Shape (T, T) @ (T, C/nheads) -> (T, C/nheads)
        assert(val.rows() == T && val.cols() == C / nheads_);
        // Stack the matrix from each head to get (T, C)
        for (int t = 0; t < T; t++) {
            memcpy(out.raw() + (t * C) + (i * C) / nheads_, val.raw() + t * (C / nheads_),
                   C / nheads_ * sizeof(float));
        }
    }
    // Apply the linear projection
    // shapes: out(T, C) @ attproj(C, C) -> (T, C)
    Linear lproj(MatrixView(&attprojw_[0], C, C), attprojb_);
    Tensor attnproj = lproj.forward(out);
    assert(attnproj.rows() == T && attnproj.cols() == C);
    return attnproj;
}

Tensor MHSA::decode(KVCache &kv_cache, int layer_index, const Tensor& x)
{
    int C = x.cols();  // For gpt-2 this is 768
    int num_tokens = kv_cache.get_num_tokens(layer_index) + 1; // include the new token
    Tensor out(1, C);  // Shape (1, C)

    // Check tensor shapes for sanity
    assert(qkvw_.size() == static_cast<size_t>(C * 3 * C));
    assert(qkvb_.size() == static_cast<size_t>(3 * C));
    assert(attprojw_.size() == static_cast<size_t>(C * C));
    assert(attprojb_.size() == static_cast<size_t>(C));

    // first step is the linear: h = x @ qkvw^T + qkvb
    // qkvw is of shape (3C, C).
    // shape: lqkv (1, 3C) <- x (1, C) @ qkvw (3C, C)^T + qkvb (3C)
    MatrixView qkvw(&qkvw_[0], 3 * C, C);

    // Now apply the linear
    Linear lqkv(qkvw, qkvb_);
    Tensor qkv = lqkv.forward(x);  // Shape (1, 3C)
    assert(qkv.rows() == 1 && qkv.cols() == 3 * C);

    Tensor keys, values;
    Tensor keys_token = Tensor(1, C), values_token = Tensor(1, C);

    kv_cache.get_keys(layer_index, keys); // Shape (T, C)
    kv_cache.get_values(layer_index, values); // Shape (T, C)

    for (int i = 0; i < C; i++)
    {
        keys.set(num_tokens - 1, i, qkv.at(0, C + i));
        values.set(num_tokens - 1, i, qkv.at(0, (2 * C) + i));
        keys_token.set(0, i, qkv.at(0, C + i));
        values_token.set(0, i, qkv.at(0, (2 * C) + i));
    }

    kv_cache.add(layer_index, keys_token, values_token);

    assert(C % nheads_ == 0);
    for (int i = 0; i < nheads_; i++) {
        // Per head q. Shape (1, C/nheads)
        MatrixView qh(qkv.raw() + (i * C) / nheads_, 1, C / nheads_, 3 * C, 1);
        // Per head k, and v. Shape (T, C/nheads)
        MatrixView kh(keys.raw() + (i * C) / nheads_, num_tokens, C / nheads_, C, 1);
        MatrixView vh(values.raw() + (i * C) / nheads_, num_tokens, C / nheads_, C, 1);
        // Shape (C/nheads, T)
        MatrixView kht = kh.transpose();

        // scaled dot product (1, C/nheads) @ (C/nheads, T) -> (1, T)
        Tensor sdp = matmul(qh, kht).scale(1.0 / sqrt(C / nheads_));  // Shape (1, T)
        assert(sdp.rows() == 1 && sdp.cols() == num_tokens);
        
        // Softmax
        Tensor softmax = sdp.softmax();  // Shape (1, T)

        // Extract the value vector
        Tensor val = softmax.matmul(vh);  // Shape (1, T) @ (T, C/nheads) -> (1, C/nheads)
        assert(val.rows() == 1 && val.cols() == C / nheads_);
        // Stack the matrix from each head to get (T, C)
        memcpy(out.raw() + (i * C) / nheads_, val.raw(),
               C / nheads_ * sizeof(float));
    }
    // Apply the linear projection
    // shapes: out(T, C) @ attproj(C, C) -> (T, C)
    Linear lproj(MatrixView(&attprojw_[0], C, C), attprojb_);
    Tensor attnproj = lproj.forward(out);
    assert(attnproj.rows() == 1 && attnproj.cols() == C);
    return attnproj;
}

Decoder::Decoder(Checkpoint& ckpt, bool use_kv_cache) {
    vocab_ = ckpt.vocab;                // vocabulary size
    nlayers_ = ckpt.layers;             // # of layers
    numheads_ = ckpt.nh;                // # of heads
    embedsize_ = ckpt.embsz;            // # embedding size
    vocab_padded_ = ckpt.vocab_padded;  // padded vocabulary size
    eps_ = 1.0e-5;                      // to prevent division by 0 in layernorm
    maxT_ = ckpt.maxT;                  // max context size
    use_kv_cache_ = use_kv_cache;
    kv_cache_ = KVCache(nlayers_, embedsize_);

    // Load weights into decoder
    for (int i = 0; i < nlayers_; i++) {
        loadLayerWeights(
            i,
            ckpt.mp.span_at<float>(ckpt.offset_ln1w + (ckpt.size_ln1w / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln1b + (ckpt.size_ln1b / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_qkvw + (ckpt.size_qkvw / nlayers_) * i,
                                   3 * embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_qkvb + (ckpt.size_qkvb / nlayers_) * i,
                                   3 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_attprojw + (ckpt.size_attprojw / nlayers_) * i,
                                   embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_attprojb + (ckpt.size_attprojb / nlayers_) * i,
                                   embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln2w + (ckpt.size_ln2w / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln2b + (ckpt.size_ln2b / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcw + (ckpt.size_fcw / nlayers_) * i,
                                   4 * embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcb + (ckpt.size_fcb / nlayers_) * i,
                                   4 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcprojw + (ckpt.size_fcprojw / nlayers_) * i,
                                   embedsize_ * 4 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcprojb + (ckpt.size_fcprojb / nlayers_) * i,
                                   embedsize_));
    }

    // Load final layernorm weights
    loadLNWeights(ckpt.mp.span_at<float>(ckpt.offset_lnfw, embedsize_),
                  ckpt.mp.span_at<float>(ckpt.offset_lnfb, embedsize_));

    // Load vocabulary projection weights
    // GPT2 uses the same weights matrix for the final vocab projection as the token embedding
    // matrix (wte)
    loadVocabProjWeights(ckpt.mp.span_at<float>(ckpt.offset_wte, vocab_padded_ * embedsize_));

    // Initialize the token embedding matrix
    // wte (vocab_padded, embsz)
    wte_ =
        MatrixView(reinterpret_cast<const float*>(ckpt.mp.bytes().subspan(ckpt.offset_wte).data()),
                   vocab_padded_, embedsize_);

    // Initialize the positional encoding matrix
    // wpe (maxT, embsz)
    wpe_ =
        MatrixView(reinterpret_cast<const float*>(ckpt.mp.bytes().subspan(ckpt.offset_wpe).data()),
                   maxT_, embedsize_);
}

// Loads weights for a layer
void Decoder::loadLayerWeights(int i, span<const float> ln1w, span<const float> ln1b,
                               span<const float> qkvw, span<const float> qkvb,
                               span<const float> attprojw, span<const float> attprojb,
                               span<const float> ln2w, span<const float> ln2b,
                               span<const float> fcw, span<const float> fcb,
                               span<const float> fcprojw, span<const float> fcprojb) {
    LayerKVCache &layer_kv_cache = kv_cache_.get_layer_kv_cache(i);
    layers_.emplace_back(use_kv_cache_, layer_kv_cache, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw,
                         fcprojb, eps_, numheads_, embedsize_);
}

void Decoder::loadLNWeights(span<const float> lnfw, span<const float> lnfb) {
    finalLN_ = LayerNorm(lnfw, lnfb, eps_);
}

void Decoder::loadVocabProjWeights(span<const float> wte) {
    vocabproj_ = Linear(MatrixView(wte.data(), vocab_padded_, embedsize_), nullopt);
}

// Looks up tokens in an embedding matrix -> (T, C)
// Adds positional embeddings -> (T, C)
// Returns a tensor of shape (T, C)
Tensor Decoder::embed(span<const int32_t> tokens) {
    int T = static_cast<int>(tokens.size());
    assert(T >= 0 && T <= wpe_.rows);
    Tensor x(T, wte_.cols);
    for (int i = 0; i < T; i++) {
        assert(tokens[i] >= 0 && tokens[i] < wte_.rows);
        for (int j = 0; j < wte_.cols; j++) {
            x.set(i, j, wte_.at(tokens[i], j) + wpe_.at(i, j));
        }
    }
    return x;
}

// Looks up tokens in an embedding matrix -> (1, C)
// Adds positional embeddings -> (1, C)
// Returns a tensor of shape (1, C)
Tensor Decoder::embed_one(const int32_t token, const int32_t position) {
    Tensor x(1, wte_.cols);
    assert(token >= 0 && token < wte_.rows);
    for (int j = 0; j < wte_.cols; j++) {
        x.set(0, j, wte_.at(token, j) + wpe_.at(position, j));
    }
    return x;
}

// The input into the decoder is of shape (T, C), and output are logits of shape (1, V)
// the predicted token
Tensor Decoder::forward(
    vector<int32_t> tokens,
    const std::function<void(const int index, const Tensor&)>& checkActivations) {
    int tensorindex = 0;
    Tensor x = embed(tokens);        // embedding + postitional encoding
    assert(x.cols() == embedsize_);  // shape of x (T, C)
    assert(x.rows() <= maxT_);       // make sure # of tokens <= max context window
    if (checkActivations) {
        checkActivations(tensorindex++, x);
    }

    // Run through all the layers
    for (int i = 0; i < nlayers_; i++) {
        x = layers_[i].forward(x);
        if (checkActivations) {
            checkActivations(tensorindex++, x);
        }
    }

    // Just grab the last position so we'll end up with (1, C)
    Tensor logits = Tensor(1, embedsize_);
    for (int i = 0; i < embedsize_; i++) {
        logits.set(0, i, x.at(x.rows() - 1, i));
    }

    // Final layerNorm, shape (1, C)
    logits = finalLN_.forward(logits);
    assert(logits.rows() == 1 && logits.cols() == embedsize_);
    if (checkActivations) {
        checkActivations(tensorindex++, logits);
    }

    // Vocabulary projection, shape (1, Vp)
    logits = vocabproj_.forward(logits);
    assert(logits.rows() == 1 && logits.cols() == vocab_padded_);
    if (checkActivations) {
        checkActivations(tensorindex++, logits);
    }

    // Remove the padded entries, shape (1, V) after this
    Tensor final_logits(1, vocab_);
    for (int i = 0; i < vocab_; i++) {
        final_logits.set(0, i, logits.at(0, i));
    }
    assert(final_logits.rows() == 1 && final_logits.cols() == vocab_);
    return final_logits;
}

/* Generate the KV cache for all the input prompt tokens by running through a forward pass 
 * of the Decoder. 
 * 
 * As an example, if the input is "Once upon a time", 
 * the prefill step will generate the Keys and Values of the tokens "Once upon a".
 * This involves a forward pass that processes all the tokens at once by processing 
 * all the 12 layers for the given T tokens and remembering the per-layer KV cache.
 * The size of the KV Cache is 2 * L * T tensors each of size 1 X C.
 * 
 * The decoder prefill step is needed only once per processing of the input tokens */
void Decoder::prefill(vector<int32_t> tokens)
{
    Tensor x = embed(tokens);        // embedding + postitional encoding
    assert(x.cols() == embedsize_);  // shape of x (T, C)
    assert(x.rows() <= maxT_);       // make sure # of tokens <= max context window

    for (int i = 0; i < nlayers_; i++) {
        x = layers_[i].forward(x);
    }
}

/* The decode step relies on the previously computed KV cache and uses it to 
 * decode the current token that is generated.
 *
 * The query vector for the token is produced. In the example, that would be "time" and 
 * the KV Cache's Key and Value tensors are used to produce a (T + 1) X C matrices for
 * K and V.
 * 
 * Then, the QK.softmax() * V produces a 1 X C tensor which is used for predicting the 
 * next token.
 * 
 * The generated K, V values for the current query token is appended into the KV cache. */
Tensor Decoder::decode(
    int32_t token, int32_t position, const std::function<void(const int index, const Tensor&)>& checkActivations) 
{
    int tensorindex = 0;

    Tensor x = embed_one(token, position);      // embedding + postitional encoding
    assert(x.cols() == embedsize_);  // shape of x (T, C)
    assert(x.rows() <= maxT_);       // make sure # of tokens <= max context window
    if (checkActivations) {
        checkActivations(tensorindex++, x);
    }

    // Run through all the layers
    for (int i = 0; i < nlayers_; i++) {
        x = layers_[i].decode(kv_cache_, i, x);
        if (checkActivations) {
            checkActivations(tensorindex++, x);
        }
    }
    
    Tensor logits = Tensor(1, embedsize_);
    // Final layerNorm, shape (1, C)
    logits = finalLN_.forward(x);
    assert(logits.rows() == 1 && logits.cols() == embedsize_);
    if (checkActivations) {
        checkActivations(tensorindex++, logits);
    }

    // Vocabulary projection, shape (1, Vp)
    logits = vocabproj_.forward(logits);
    assert(logits.rows() == 1 && logits.cols() == vocab_padded_);
    if (checkActivations) {
        checkActivations(tensorindex++, logits);
    }

    // Remove the padded entries, shape (1, V) after this
    Tensor final_logits(1, vocab_);
    for (int i = 0; i < vocab_; i++) {
        final_logits.set(0, i, logits.at(0, i));
    }
    assert(final_logits.rows() == 1 && final_logits.cols() == vocab_);
    return final_logits;
}

void Decoder::forget_all_tokens()
{
    int num_tokens = kv_cache_.get_num_tokens(0);

    for (int i = 0; i < num_tokens; i++)
    {
        kv_cache_.remove_oldest_token();
    }
}