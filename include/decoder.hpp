#pragma once

#include <limits.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "checkpoint.hpp"
#include "kvcache.hpp"
#include "tensor.hpp"

using kvcache::KVCache;
using kvcache::LayerKVCache;
using tn::MatrixView;
using tn::Tensor;

namespace decoder {

class Linear {
   public:
    Linear(MatrixView w, std::optional<std::span<const float>> b) : w_(w), b_(b) {}
    Linear() = default;
    Tensor forward(const Tensor& x) const;

   private:
    MatrixView w_;
    std::optional<std::span<const float>> b_;
};

class LayerNorm {
   public:
    LayerNorm() = default;
    LayerNorm(std::span<const float> w, std::span<const float> b, float eps)
        : w_(w), b_(b), eps_(eps) {}
    Tensor forward(const Tensor& x) const;

   private:
    std::span<const float> w_;
    std::span<const float> b_;
    float eps_;
};

class MHSA {
   public:
    MHSA(bool use_kv_cache, LayerKVCache& layer_kv_cache, std::span<const float> qkvw,
         std::span<const float> qkvb, std::span<const float> attprojw,
         std::span<const float> attprojb, int nheads)
        : use_kv_cache_(use_kv_cache),
          kv_cache_(layer_kv_cache),
          qkvw_(qkvw),
          qkvb_(qkvb),
          attprojw_(attprojw),
          attprojb_(attprojb),
          nheads_(nheads) {}
    MHSA() = delete;
    Tensor forward(const Tensor& x) const;
    Tensor decode(KVCache& kv_cache, int layer_index, const Tensor& x);

   private:
    bool use_kv_cache_;
    LayerKVCache& kv_cache_;
    std::span<const float> qkvw_;      // shape (3C, C) C is the embedding dimension
    std::span<const float> qkvb_;      // shape (3C)
    std::span<const float> attprojw_;  // shape (C, C)
    std::span<const float> attprojb_;  // shape (C)
    int nheads_;                       // number of attention heads
};

class FFN {
   public:
    FFN(MatrixView w1, std::span<const float> b1, MatrixView wproj, std::span<const float> bproj)
        : l1_(w1, b1), l2_(wproj, bproj) {}
    FFN() = default;
    Tensor forward(const Tensor& x) const;

   private:
    Linear l1_;
    Linear l2_;
};

class Layer {
   public:
    Layer() = delete;
    Layer(bool use_kv_cache, LayerKVCache& layer_kv_cache, std::span<const float> ln1w,
          std::span<const float> ln1b, std::span<const float> qkvw, std::span<const float> qkvb,
          std::span<const float> attprojw, std::span<const float> attprojb,
          std::span<const float> ln2w, std::span<const float> ln2b, std::span<const float> fcw,
          std::span<const float> fcb, std::span<const float> fcprojw,
          std::span<const float> fcprojb, float eps, int nheads, int embsz)
        : ln1_(LayerNorm(ln1w, ln1b, eps)),
          attn_(MHSA(use_kv_cache, layer_kv_cache, qkvw, qkvb, attprojw, attprojb, nheads)),
          ln2_(LayerNorm(ln2w, ln2b, eps)),
          ffn_(FFN(MatrixView(fcw.data(), 4 * embsz, embsz), fcb,
                   MatrixView(fcprojw.data(), embsz, 4 * embsz), fcprojb)) {}
    Tensor forward(const Tensor& x);
    Tensor decode(KVCache& kv_cache, int layer_index, const Tensor& x);

   private:
    LayerNorm ln1_;
    MHSA attn_;
    LayerNorm ln2_;
    FFN ffn_;

    Tensor run(const Tensor& x, KVCache* kv_cache, int layer_index);
};

class Decoder {
   public:
    Decoder(Checkpoint& ckpt, bool use_kv_cache);
    Tensor forward(std::vector<int32_t> tokens,
                   const std::function<void(const int index, const Tensor&)>& checkActivations);
    Tensor embed(std::span<const int32_t> tokens);
    Tensor embed_one(const int32_t token, const int32_t position);
    void prefill(std::vector<int32_t> tokens);
    Tensor decode(int32_t token, int32_t position,
                  const std::function<void(const int index, const Tensor&)>& checkActivations);
    void forget_all_tokens();

   private:
    int vocab_;                  // vocabulary size
    int nlayers_;                // # of layers
    int numheads_;               // # of heads
    int embedsize_;              // # embedding size
    int vocab_padded_;           // padded vocabulary size
    std::vector<Layer> layers_;  // each of the layers
    LayerNorm finalLN_;          // final layer norm, outside all the 12 layers
    Linear vocabproj_;           // final vocabulary projection linear
    float eps_;                  // to prevent division by 0 in layernorm
    int maxT_;                   // max context size
    MatrixView wte_;             // token embeding matrix
    MatrixView wpe_;             // positional encoding matrix
    bool use_kv_cache_;          // should the decoder use key value cache?
    KVCache kv_cache_;           // key-value cache

    Tensor run(Tensor x, bool cached,
               const std::function<void(const int index, const Tensor&)>& checkActivations);
    void loadFromCheckpoint(Checkpoint checkpoint);
    void loadLayerWeights(int i, std::span<const float> ln1w, std::span<const float> ln1b,
                          std::span<const float> qkvw, std::span<const float> qkvb,
                          std::span<const float> attprojw, std::span<const float> attprojb,
                          std::span<const float> ln2w, std::span<const float> ln2b,
                          std::span<const float> fcw, std::span<const float> fcb,
                          std::span<const float> fcprojw, std::span<const float> fcprojb);
    void loadLNWeights(std::span<const float> lnfw, std::span<const float> lnfb);
    void loadVocabProjWeights(std::span<const float> wte);
};

}  // namespace decoder